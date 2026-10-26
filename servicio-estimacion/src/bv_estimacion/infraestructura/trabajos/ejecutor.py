"""Workers que toman trabajos de la cola y los procesan, y el barrido periódico de lo vencido."""

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

import anyio

from bv_estimacion.aplicacion.puertos import ColaTrabajos
from bv_estimacion.dominio.valores import IdTrabajo

logger = logging.getLogger(__name__)


class Procesador(Protocol):
    async def ejecutar(self, id_trabajo: IdTrabajo) -> None: ...


class EjecutorTrabajos:
    """Corre `cantidad` workers hasta que lo cancelen. Pensado para lanzarse desde el `lifespan`:

        async with anyio.create_task_group() as grupo:
            grupo.start_soon(ejecutor.correr)
            yield
            grupo.cancel_scope.cancel()   # apagado: cada worker corta su espera o su trabajo en curso

    Un error en un trabajo se registra y el worker sigue con el siguiente: **un worker nunca muere por un error**.
    La cancelación sí lo detiene (es `BaseException`, no se atrapa).
    """

    def __init__(self, cola: ColaTrabajos, procesador: Procesador, cantidad: int) -> None:
        if cantidad < 1:
            raise ValueError("hace falta al menos un worker")
        self._cola = cola
        self._procesador = procesador
        self._cantidad = cantidad

    async def correr(self) -> None:
        async with anyio.create_task_group() as grupo:
            for _ in range(self._cantidad):
                grupo.start_soon(self._worker)

    async def _worker(self) -> None:
        while True:
            id_trabajo = await self._cola.siguiente()
            try:
                await self._procesador.ejecutar(id_trabajo)
            except Exception:
                logger.exception("worker_error", extra={"id_trabajo": str(id_trabajo)})


Barrido = Callable[[], Awaitable[int]]


async def barrer_periodicamente(intervalo_s: float, barridos: Sequence[Barrido]) -> None:
    """Corre cada barrido (por ejemplo, `purgar_vencidos` y `barrer_vencidos`) cada `intervalo_s` segundos, hasta
    que lo cancelen. Un barrido que falla se registra y no corta a los demás ni a los siguientes ciclos."""
    while True:
        await anyio.sleep(intervalo_s)
        for barrido in barridos:
            try:
                borrados = await barrido()
            except Exception:
                logger.exception("barrido_error")
            else:
                if borrados:
                    logger.info("barrido", extra={"borrados": borrados})
