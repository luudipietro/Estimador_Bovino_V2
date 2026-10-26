"""RF-19: recibir un pedido de estimación y dejarlo en cola, sin esperar la inferencia."""

import hashlib
import logging
from dataclasses import dataclass, field

from bv_estimacion.aplicacion.dto import SolicitarEstimacionCmd, SolicitudAceptada
from bv_estimacion.aplicacion.puertos import (
    AlmacenTemporalImagenes,
    ColaLlena,
    ColaTrabajos,
    GeneradorIds,
    Reloj,
    RepositorioTrabajos,
)
from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio
from bv_estimacion.dominio.estimacion import MAX_IMAGENES, MIN_IMAGENES, Trabajo, huella_de_imagenes
from bv_estimacion.dominio.valores import IdTrabajo, OrigenImagen, ReferenciaImagen

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Compensacion:
    """Lo que ya se persistió y hay que deshacer si el pedido no llega a quedar en la cola."""

    ubicaciones: list[str] = field(default_factory=list)
    id_trabajo: IdTrabajo | None = None


class SolicitarEstimacion:
    """Orden fijo, para no dejar trabajos huérfanos:

    1. buscar un trabajo existente para ese `id_evento` (idempotencia);
    2. chequear que haya lugar en la cola: si no, se rechaza **antes de persistir nada**;
    3. guardar las imágenes (antes de responder 202);
    4. guardar el trabajo;
    5. encolar, **lo último**. Si falla, se revierte todo lo anterior.
    """

    def __init__(
        self,
        trabajos: RepositorioTrabajos,
        cola: ColaTrabajos,
        almacen: AlmacenTemporalImagenes,
        ids: GeneradorIds,
        reloj: Reloj,
    ) -> None:
        self._trabajos = trabajos
        self._cola = cola
        self._almacen = almacen
        self._ids = ids
        self._reloj = reloj

    async def ejecutar(self, cmd: SolicitarEstimacionCmd) -> SolicitudAceptada:
        total = len(cmd.imagenes) + len(cmd.referencias)
        if (cmd.imagenes and cmd.referencias) or not MIN_IMAGENES <= total <= MAX_IMAGENES:
            raise ErrorDeDominio(
                CodigoError.CANTIDAD_IMAGENES_INVALIDA,
                f"se esperaban entre {MIN_IMAGENES} y {MAX_IMAGENES} imágenes, por multipart o por referencia, "
                "pero no las dos formas a la vez",
                regla="R-18",
            )

        # El SHA-256 identifica el contenido. Con multipart lo calculamos; con URL lo declara el backend.
        # `ReferenciaImagen` valida que esté bien formado (la API ya lo rechazó antes si no).
        externas = [ReferenciaImagen(OrigenImagen.URL, r.url, r.sha256) for r in cmd.referencias]
        subidas = [(contenido, hashlib.sha256(contenido).hexdigest()) for contenido in cmd.imagenes]
        huella = huella_de_imagenes([h for _, h in subidas] or [r.sha256 for r in externas])

        existente = await self._trabajos.buscar_por_id_evento(cmd.id_evento)
        if existente is not None:
            return self._resolver_existente(existente, huella)

        if not await self._cola.hay_lugar():
            raise ErrorDeDominio(CodigoError.COLA_LLENA, "la cola de trabajos está llena")

        compensacion = _Compensacion()
        try:
            return await self._crear(cmd, subidas, externas, huella, compensacion)
        except ColaLlena as error:  # otro pedido ocupó el último lugar entre `hay_lugar` y `encolar`
            await self._deshacer(compensacion)
            raise ErrorDeDominio(CodigoError.COLA_LLENA, "la cola de trabajos está llena") from error
        except Exception:
            await self._deshacer(compensacion)
            raise

    async def _crear(
        self,
        cmd: SolicitarEstimacionCmd,
        subidas: list[tuple[bytes, str]],
        externas: list[ReferenciaImagen],
        huella: str,
        compensacion: _Compensacion,
    ) -> SolicitudAceptada:
        referencias = list(externas)
        for contenido, sha256 in subidas:
            ubicacion = await self._almacen.guardar(contenido)
            compensacion.ubicaciones.append(ubicacion)
            referencias.append(ReferenciaImagen(OrigenImagen.ALMACEN_TEMPORAL, ubicacion, sha256))

        trabajo = Trabajo.crear(
            self._ids.nuevo_id_trabajo(),
            cmd.id_evento,
            referencias,
            self._reloj.ahora(),
            metadatos=cmd.metadatos,
            url_notificacion=cmd.url_notificacion,
        )
        guardado = await self._trabajos.agregar(trabajo)
        if guardado.id != trabajo.id:
            # Otro pedido con el mismo `id_evento` ganó la carrera: lo guardado acá no se va a usar.
            await self._deshacer(compensacion)
            compensacion.ubicaciones.clear()
            return self._resolver_existente(guardado, huella)

        compensacion.id_trabajo = trabajo.id
        await self._cola.encolar(trabajo.id)
        return SolicitudAceptada(trabajo=trabajo, nuevo=True)

    @staticmethod
    def _resolver_existente(existente: Trabajo, huella: str) -> SolicitudAceptada:
        """Mismo `id_evento` + misma huella (aunque las imágenes vengan en otro orden) = mismo trabajo.
        Con otra huella es un conflicto: el trabajo original no se toca."""
        if existente.es_mismo_pedido(huella):
            return SolicitudAceptada(trabajo=existente, nuevo=False)
        raise ErrorDeDominio(
            CodigoError.ID_EVENTO_EN_CONFLICTO,
            "ese id_evento ya se usó con otras imágenes",
        )

    async def _deshacer(self, compensacion: _Compensacion) -> None:
        """Revierte lo persistido. Es de mejor esfuerzo: un fallo acá no tapa el error original."""
        if compensacion.id_trabajo is not None:
            try:
                await self._trabajos.eliminar(compensacion.id_trabajo)
            except Exception:
                logger.exception(
                    "no se pudo eliminar el trabajo al revertir",
                    extra={"id_trabajo": str(compensacion.id_trabajo)},
                )
        for ubicacion in compensacion.ubicaciones:
            try:
                await self._almacen.borrar(ubicacion)
            except Exception:
                logger.exception("no se pudo borrar una imagen temporal al revertir")
