"""RF-20, RF-22, RNF-01: procesar un trabajo encolado."""

import hashlib
import logging
from collections.abc import Sequence
from datetime import datetime

import anyio

from bv_estimacion.aplicacion.puertos import (
    AlmacenTemporalImagenes,
    ImagenNoDisponible,
    ModeloNoDisponible,
    ModeloPeso,
    Notificador,
    ObtenedorImagen,
    Reloj,
    RepositorioTrabajos,
)
from bv_estimacion.dominio.errores import CodigoError
from bv_estimacion.dominio.estimacion import EstadoTrabajo, ResultadoFoto, Trabajo, agregar_resultados
from bv_estimacion.dominio.valores import IdTrabajo, OrigenImagen, ReferenciaImagen

logger = logging.getLogger(__name__)


def _ms(desde: datetime, hasta: datetime) -> int:
    return round((hasta - desde).total_seconds() * 1000)


class ProcesarEstimacion:
    """Toma un trabajo `pendiente`, estima sus fotos y lo deja `completado` o `fallido`.

    - **El "no sé" no es un error (R-21):** una foto no confiable termina `completado` con `confiable: false`.
    - **Presupuesto de tiempo (RNF-01):** descarga e inferencia juntas no pueden pasar `presupuesto_s`. Si se pasa,
      el trabajo queda `fallido` con `tiempo_excedido`. La espera por un lugar en el pool del modelo también
      cuenta: si un hilo quedó colgado de un trabajo anterior (ver el contrato de `ModeloPeso.estimar`), los
      siguientes pueden vencer esperándolo.
    - **Un fallo fuera de la estimación (por ejemplo, al guardar)** deja el trabajo `fallido` con `error_interno`.
      Es un mejor esfuerzo: si el repositorio sigue sin responder, el segundo guardado también falla y solo se
      registra. Un trabajo cancelado por el apagado queda como estaba.
    - **El aviso al backend va después** de guardar el estado final y **fuera del presupuesto**.
    - **Las imágenes temporales se borran siempre**, sea cual sea el final, incluso si el worker se cancela.
    """

    def __init__(
        self,
        modelo: ModeloPeso,
        trabajos: RepositorioTrabajos,
        almacen: AlmacenTemporalImagenes,
        obtenedor: ObtenedorImagen,
        notificador: Notificador,
        reloj: Reloj,
        presupuesto_s: float,
    ) -> None:
        self._modelo = modelo
        self._trabajos = trabajos
        self._almacen = almacen
        self._obtenedor = obtenedor
        self._notificador = notificador
        self._reloj = reloj
        self._presupuesto_s = presupuesto_s

    async def ejecutar(self, id_trabajo: IdTrabajo) -> None:
        trabajo = await self._trabajos.obtener(id_trabajo)
        if trabajo is None:
            logger.warning("trabajo_inexistente", extra={"id_trabajo": str(id_trabajo)})
            return
        if trabajo.estado is not EstadoTrabajo.PENDIENTE:
            # Un trabajo no se procesa dos veces (por ejemplo, si la cola entrega un mensaje repetido).
            logger.info("trabajo_ya_procesado", extra={"id_trabajo": str(id_trabajo), "estado": trabajo.estado.value})
            return

        trabajo = trabajo.iniciar(self._reloj.ahora())

        try:
            try:
                await self._trabajos.guardar(trabajo)
                final = await self._resolver(trabajo)
            finally:
                with anyio.CancelScope(shield=True):  # que se borren aunque nos estén cancelando
                    await self._borrar_temporales(trabajo.imagenes)
            await self._trabajos.guardar(final)
        except Exception:
            # Falló algo fuera de la estimación (típicamente el repositorio). Se intenta que el trabajo no quede
            # `procesando` para siempre; si el repositorio sigue caído, no hay nada más que hacer acá.
            logger.exception("error_al_procesar", extra={"id_trabajo": str(id_trabajo)})
            final = self._fallar(trabajo, CodigoError.ERROR_INTERNO, reintentable=False)
            try:
                await self._trabajos.guardar(final)
            except Exception:
                logger.exception("no_se_pudo_marcar_fallido", extra={"id_trabajo": str(id_trabajo)})
                return

        logger.info(
            "trabajo_finalizado",
            extra={"id_trabajo": str(final.id), "id_evento": str(final.id_evento), "estado": final.estado.value},
        )
        try:
            await self._notificador.notificar(final)
        except Exception:
            # El resultado ya está guardado y se consulta por GET: no se puede perder por esto.
            logger.exception(
                "no_se_pudo_agendar_la_notificacion",
                extra={"id_trabajo": str(final.id), "id_evento": str(final.id_evento)},
            )

    async def _resolver(self, trabajo: Trabajo) -> Trabajo:
        """Devuelve el trabajo en su estado final. Los fallos esperables se traducen a un código del catálogo."""
        try:
            with anyio.fail_after(self._presupuesto_s):
                fotos, advertencias, descarga_ms, inferencia_ms = await self._estimar_fotos(trabajo)
            resultado = agregar_resultados(fotos, self._modelo.version, advertencias)
            return trabajo.completar(
                resultado,
                self._reloj.ahora(),
                descarga_ms=descarga_ms,
                inferencia_ms=inferencia_ms,
            )
        except TimeoutError:
            return self._fallar(trabajo, CodigoError.TIEMPO_EXCEDIDO, reintentable=True)
        except ImagenNoDisponible as error:
            logger.warning("imagen_no_disponible", extra={"id_evento": str(trabajo.id_evento), "motivo": str(error)})
            return self._fallar(trabajo, CodigoError.REFERENCIA_INACCESIBLE, reintentable=True)
        except ModeloNoDisponible:
            return self._fallar(trabajo, CodigoError.MODELO_NO_DISPONIBLE, reintentable=True)
        except Exception:
            # Detalle al log; el cliente solo ve `error_interno`, sin trazas ni nombres de clases.
            logger.exception("error_inesperado", extra={"id_evento": str(trabajo.id_evento)})
            return self._fallar(trabajo, CodigoError.ERROR_INTERNO, reintentable=False)

    def _fallar(self, trabajo: Trabajo, codigo: CodigoError, *, reintentable: bool) -> Trabajo:
        return trabajo.fallar(codigo, self._reloj.ahora(), reintentable=reintentable)

    async def _estimar_fotos(self, trabajo: Trabajo) -> tuple[list[ResultadoFoto], tuple[str, ...], int, int]:
        fotos: list[ResultadoFoto] = []
        advertencias: list[str] = []
        descarga_ms = 0
        inferencia_ms = 0
        # Los índices siguen el orden del primer pedido, que es el que guarda el trabajo.
        for indice, referencia in enumerate(trabajo.imagenes, start=1):
            t0 = self._reloj.ahora()
            contenido = await self._leer(referencia)
            t1 = self._reloj.ahora()
            estimacion = await self._modelo.estimar(contenido)
            t2 = self._reloj.ahora()
            descarga_ms += _ms(t0, t1)
            inferencia_ms += _ms(t1, t2)
            fotos.append(estimacion.resultado)
            advertencias.extend(f"la imagen {indice} {aviso}" for aviso in estimacion.advertencias)
        return fotos, tuple(advertencias), descarga_ms, inferencia_ms

    async def _leer(self, referencia: ReferenciaImagen) -> bytes:
        """Lee la imagen y verifica que sea la declarada. Si no está o no coincide, `ImagenNoDisponible`."""
        if referencia.origen is OrigenImagen.ALMACEN_TEMPORAL:
            contenido = await self._almacen.obtener(referencia.ubicacion)
            if contenido is None:
                raise ImagenNoDisponible("la imagen ya no está en el almacén temporal")
        else:
            contenido = await self._obtenedor.obtener(referencia.ubicacion)
        if hashlib.sha256(contenido).hexdigest() != referencia.sha256:
            raise ImagenNoDisponible("el contenido no coincide con el sha256 declarado")
        return contenido

    async def _borrar_temporales(self, referencias: Sequence[ReferenciaImagen]) -> None:
        for referencia in referencias:
            if referencia.origen is not OrigenImagen.ALMACEN_TEMPORAL:
                continue
            try:
                await self._almacen.borrar(referencia.ubicacion)
            except Exception:
                # El TTL de respaldo del almacén la va a limpiar.
                logger.exception("no_se_pudo_borrar_una_imagen_temporal")
