"""Repositorio y cola de trabajos en memoria (desarrollo, tests y la primera homologación).

Redis queda para más adelante, detrás de estos mismos puertos, y pendiente de que el equipo lo confirme.
Todo se pierde si el proceso se reinicia: el backend lo nota por el `404` y reenvía con el mismo `id_evento`.
"""

import asyncio
from datetime import datetime, timedelta

from bv_estimacion.aplicacion.puertos import ColaLlena, Reloj
from bv_estimacion.dominio.estimacion import Trabajo
from bv_estimacion.dominio.valores import IdEvento, IdTrabajo


class RepositorioTrabajosMemoria:
    """Un trabajo vive `ttl_s` segundos **desde que termina**. Mientras está pendiente o procesando no vence.

    Es seguro dentro del event loop: ningún método cede el control entre mirar y escribir, así que `agregar` es
    atómico por `id_evento` sin necesitar un lock.
    """

    def __init__(self, ttl_s: float, reloj: Reloj) -> None:
        self._ttl = timedelta(seconds=ttl_s)
        self._reloj = reloj
        self._trabajos: dict[IdTrabajo, Trabajo] = {}
        self._por_id_evento: dict[IdEvento, IdTrabajo] = {}
        self._vence: dict[IdTrabajo, datetime] = {}

    def __len__(self) -> int:
        return len(self._trabajos)

    # --- puerto RepositorioTrabajos -------------------------------------------------------------

    async def agregar(self, trabajo: Trabajo) -> Trabajo:
        existente = self._vigente_por_id_evento(trabajo.id_evento)
        if existente is not None:
            return existente
        self._trabajos[trabajo.id] = trabajo
        self._por_id_evento[trabajo.id_evento] = trabajo.id
        self._registrar_vencimiento(trabajo)
        return trabajo

    async def guardar(self, trabajo: Trabajo) -> None:
        self._trabajos[trabajo.id] = trabajo
        self._por_id_evento[trabajo.id_evento] = trabajo.id
        self._registrar_vencimiento(trabajo)

    async def obtener(self, id_trabajo: IdTrabajo) -> Trabajo | None:
        self._purgar_si_vencio(id_trabajo)
        return self._trabajos.get(id_trabajo)

    async def buscar_por_id_evento(self, id_evento: IdEvento) -> Trabajo | None:
        return self._vigente_por_id_evento(id_evento)

    async def eliminar(self, id_trabajo: IdTrabajo) -> None:
        self._borrar(id_trabajo)

    # --- mantenimiento --------------------------------------------------------------------------

    async def purgar_vencidos(self) -> int:
        """Barrido completo. Devuelve cuántos trabajos borró. Se programa de forma periódica."""
        ahora = self._reloj.ahora()
        vencidos = [id_trabajo for id_trabajo, vence in self._vence.items() if vence <= ahora]
        for id_trabajo in vencidos:
            self._borrar(id_trabajo)
        return len(vencidos)

    # --- internos -------------------------------------------------------------------------------

    def _registrar_vencimiento(self, trabajo: Trabajo) -> None:
        if trabajo.finalizado is not None:
            self._vence[trabajo.id] = trabajo.finalizado + self._ttl

    def _purgar_si_vencio(self, id_trabajo: IdTrabajo) -> None:
        vence = self._vence.get(id_trabajo)
        if vence is not None and vence <= self._reloj.ahora():
            self._borrar(id_trabajo)

    def _vigente_por_id_evento(self, id_evento: IdEvento) -> Trabajo | None:
        id_trabajo = self._por_id_evento.get(id_evento)
        if id_trabajo is None:
            return None
        self._purgar_si_vencio(id_trabajo)
        return self._trabajos.get(id_trabajo)

    def _borrar(self, id_trabajo: IdTrabajo) -> None:
        trabajo = self._trabajos.pop(id_trabajo, None)
        self._vence.pop(id_trabajo, None)
        if trabajo is not None and self._por_id_evento.get(trabajo.id_evento) == id_trabajo:
            del self._por_id_evento[trabajo.id_evento]


class ColaTrabajosMemoria:
    def __init__(self, capacidad: int) -> None:
        self._cola: asyncio.Queue[IdTrabajo] = asyncio.Queue(maxsize=capacidad)

    async def hay_lugar(self) -> bool:
        return not self._cola.full()

    async def encolar(self, id_trabajo: IdTrabajo) -> None:
        try:
            self._cola.put_nowait(id_trabajo)
        except asyncio.QueueFull as error:
            raise ColaLlena from error

    async def siguiente(self) -> IdTrabajo:
        return await self._cola.get()

    def __len__(self) -> int:
        return self._cola.qsize()
