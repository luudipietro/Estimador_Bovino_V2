"""Fakes deterministas de los puertos de `aplicacion/`. Sin red, sin GPU y sin los `.pt`."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import anyio

from bv_estimacion.aplicacion.dto import EstimacionFoto
from bv_estimacion.aplicacion.puertos import CallbackFallido, ColaLlena, ImagenNoDisponible
from bv_estimacion.dominio.estimacion import ResultadoFoto, Trabajo
from bv_estimacion.dominio.valores import IdEvento, IdTrabajo, VersionModelo

T0 = datetime(2026, 10, 25, 12, 0, 0, tzinfo=UTC)
VERSION = VersionModelo("v-test", 36.0, 621.0)


def foto_ok(peso_kg: float = 187.4, *advertencias: str) -> EstimacionFoto:
    return EstimacionFoto(ResultadoFoto(peso_kg=peso_kg, confiable=True), advertencias)


class Bitacora:
    """Registra en qué orden se llamó a los puertos de persistencia."""

    def __init__(self) -> None:
        self.eventos: list[str] = []

    def registrar(self, evento: str) -> None:
        self.eventos.append(evento)


class RelojFijo:
    def __init__(self, inicio: datetime = T0) -> None:
        self._ahora = inicio

    def ahora(self) -> datetime:
        return self._ahora

    def avanzar(self, *, segundos: float = 0.0) -> None:
        self._ahora += timedelta(seconds=segundos)


class GeneradorIdsSecuencial:
    def __init__(self) -> None:
        self._n = 0

    def nuevo_id_trabajo(self) -> IdTrabajo:
        self._n += 1
        return IdTrabajo(f"trb_{self._n:02d}")


class RepositorioEnMemoria:
    def __init__(self, bitacora: Bitacora | None = None) -> None:
        self.trabajos: dict[IdTrabajo, Trabajo] = {}
        self._bitacora = bitacora
        # Para simular una carrera: se ejecuta justo antes de `agregar`, como si otro pedido se hubiera colado.
        self.antes_de_agregar: Callable[[], None] | None = None
        self.error_al_eliminar: Exception | None = None
        # Número de llamada a `guardar` (desde 1) -> error que lanza en esa llamada.
        self.errores_al_guardar: dict[int, Exception] = {}
        self._llamadas_a_guardar = 0

    def _registrar(self, evento: str) -> None:
        if self._bitacora is not None:
            self._bitacora.registrar(evento)

    async def agregar(self, trabajo: Trabajo) -> Trabajo:
        self._registrar("repo.agregar")
        if self.antes_de_agregar is not None:
            self.antes_de_agregar()
        existente = self._por_id_evento(trabajo.id_evento)
        if existente is not None:
            return existente
        self.trabajos[trabajo.id] = trabajo
        return trabajo

    async def guardar(self, trabajo: Trabajo) -> None:
        self._registrar("repo.guardar")
        self._llamadas_a_guardar += 1
        error = self.errores_al_guardar.get(self._llamadas_a_guardar)
        if error is not None:
            raise error
        self.trabajos[trabajo.id] = trabajo

    async def obtener(self, id_trabajo: IdTrabajo) -> Trabajo | None:
        return self.trabajos.get(id_trabajo)

    async def buscar_por_id_evento(self, id_evento: IdEvento) -> Trabajo | None:
        return self._por_id_evento(id_evento)

    async def eliminar(self, id_trabajo: IdTrabajo) -> None:
        self._registrar("repo.eliminar")
        if self.error_al_eliminar is not None:
            raise self.error_al_eliminar
        self.trabajos.pop(id_trabajo, None)

    def insertar(self, trabajo: Trabajo) -> None:
        """Para preparar el estado de un test sin pasar por `agregar`."""
        self.trabajos[trabajo.id] = trabajo

    def _por_id_evento(self, id_evento: IdEvento) -> Trabajo | None:
        return next((t for t in self.trabajos.values() if t.id_evento == id_evento), None)


class ColaEnMemoria:
    def __init__(self, capacidad: int = 10, bitacora: Bitacora | None = None) -> None:
        self.capacidad = capacidad
        self.ids: list[IdTrabajo] = []
        self._bitacora = bitacora
        self.error_al_encolar: Exception | None = None

    async def hay_lugar(self) -> bool:
        if self._bitacora is not None:
            self._bitacora.registrar("cola.hay_lugar")
        return len(self.ids) < self.capacidad

    async def encolar(self, id_trabajo: IdTrabajo) -> None:
        if self._bitacora is not None:
            self._bitacora.registrar("cola.encolar")
        if self.error_al_encolar is not None:
            raise self.error_al_encolar
        if len(self.ids) >= self.capacidad:
            raise ColaLlena
        self.ids.append(id_trabajo)

    async def siguiente(self) -> IdTrabajo:
        return self.ids.pop(0)


class AlmacenTemporalEnMemoria:
    def __init__(self, bitacora: Bitacora | None = None) -> None:
        self.objetos: dict[str, bytes] = {}
        self._n = 0
        self._bitacora = bitacora
        self.fallar_en_el_guardado_numero: int | None = None  # 1 = el primero
        self.error_al_borrar: Exception | None = None

    async def guardar(self, contenido: bytes) -> str:
        if self._bitacora is not None:
            self._bitacora.registrar("almacen.guardar")
        self._n += 1
        if self.fallar_en_el_guardado_numero == self._n:
            raise OSError("disco lleno")
        referencia = f"tmp-{self._n}"
        self.objetos[referencia] = contenido
        return referencia

    async def obtener(self, referencia: str) -> bytes | None:
        return self.objetos.get(referencia)

    async def borrar(self, referencia: str) -> None:
        if self.error_al_borrar is not None:
            raise self.error_al_borrar
        self.objetos.pop(referencia, None)


class ObtenedorImagenFalso:
    def __init__(self, contenidos: dict[str, bytes] | None = None) -> None:
        self.contenidos = contenidos or {}
        self.pedidos: list[str] = []

    async def obtener(self, url: str) -> bytes:
        self.pedidos.append(url)
        if url not in self.contenidos:
            raise ImagenNoDisponible("la descarga falló")
        return self.contenidos[url]


class ModeloFalso:
    """Devuelve lo configurado según el contenido de la foto."""

    def __init__(self, reloj: RelojFijo | None = None, version: VersionModelo = VERSION) -> None:
        self._version = version
        self._reloj = reloj
        self.respuestas: dict[bytes, EstimacionFoto] = {}
        self.por_defecto = foto_ok()
        self.error: Exception | None = None
        self.espera_real_s = 0.0  # tiempo de pared: para probar el presupuesto
        self.avanza_reloj_s = 0.0  # tiempo del reloj falso: para probar las duraciones
        self.llamadas: list[bytes] = []

    @property
    def version(self) -> VersionModelo:
        return self._version

    async def estimar(self, contenido: bytes) -> EstimacionFoto:
        self.llamadas.append(contenido)
        if self.error is not None:
            raise self.error
        if self.espera_real_s:
            await anyio.sleep(self.espera_real_s)
        if self._reloj is not None:
            self._reloj.avanzar(segundos=self.avanza_reloj_s)
        return self.respuestas.get(contenido, self.por_defecto)


class ModeloConPoolDeUnSlot:
    """Modela el contrato de `ModeloPeso.estimar`: cancelar la espera NO libera el lugar del pool; lo libera el
    fin real del "hilo". Mientras `terminar` no se active, el hilo queda corriendo aunque nadie lo espere."""

    def __init__(self, version: VersionModelo = VERSION) -> None:
        self._version = version
        self._slot = asyncio.Semaphore(1)
        self.terminar = asyncio.Event()
        self.hilos_activos = 0
        self.maximo_de_hilos_activos = 0

    @property
    def version(self) -> VersionModelo:
        return self._version

    async def estimar(self, contenido: bytes) -> EstimacionFoto:
        await self._slot.acquire()  # esperar el turno sí se puede cancelar
        hilo = asyncio.ensure_future(self._hilo())
        return await asyncio.shield(hilo)  # cancelar la espera no mata al hilo

    async def _hilo(self) -> EstimacionFoto:
        self.hilos_activos += 1
        self.maximo_de_hilos_activos = max(self.maximo_de_hilos_activos, self.hilos_activos)
        try:
            await self.terminar.wait()
            return foto_ok()
        finally:
            self.hilos_activos -= 1
            self._slot.release()  # recién ahora queda libre el lugar


class NotificadorEspia:
    def __init__(self, repo: RepositorioEnMemoria | None = None) -> None:
        self.notificados: list[Trabajo] = []
        self.estado_guardado_al_notificar: list[str | None] = []
        self.error: Exception | None = None
        self.espera_real_s = 0.0
        self._repo = repo

    async def notificar(self, trabajo: Trabajo) -> None:
        if self.espera_real_s:
            await anyio.sleep(self.espera_real_s)
        if self.error is not None:
            raise self.error
        self.notificados.append(trabajo)
        if self._repo is not None:
            guardado = self._repo.trabajos.get(trabajo.id)
            self.estado_guardado_al_notificar.append(guardado.estado.value if guardado else None)


class EnviadorCallbackFalso:
    def __init__(self, fallos_antes_de_entregar: int = 0) -> None:
        self.fallos_pendientes = fallos_antes_de_entregar
        self.intentos: list[str] = []
        self.error_inesperado: Exception | None = None

    async def enviar(self, url: str, trabajo: Trabajo) -> None:
        self.intentos.append(url)
        if self.error_inesperado is not None:
            raise self.error_inesperado
        if self.fallos_pendientes > 0:
            self.fallos_pendientes -= 1
            raise CallbackFallido("el backend no respondió")


class DormidorFalso:
    def __init__(self) -> None:
        self.esperas: list[float] = []

    async def dormir(self, segundos: float) -> None:
        self.esperas.append(segundos)
