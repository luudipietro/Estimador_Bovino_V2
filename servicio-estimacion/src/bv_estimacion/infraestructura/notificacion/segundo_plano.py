"""Notificador que agenda el aviso en una tarea aparte (implementa `Notificador`)."""

import logging
from collections.abc import Awaitable, Callable

from anyio.abc import TaskGroup

from bv_estimacion.dominio.estimacion import Trabajo

logger = logging.getLogger(__name__)


class NotificadorEnSegundoPlano:
    """`notificar` agenda la entrega y **vuelve enseguida**: ni los reintentos ni la espera del backend retrasan al
    worker, ocupan el lugar del modelo o cuentan dentro del presupuesto de tiempo.

    Las tareas viven en el `TaskGroup` que arma el `lifespan`, así que al apagar la app se cancelan en orden. Una
    tarea que falla **no puede cancelar al grupo** (llevaría con ella a los workers): por eso se protege acá.
    """

    def __init__(self, entregar: Callable[[Trabajo], Awaitable[bool]], grupo: TaskGroup) -> None:
        self._entregar = entregar  # `NotificarResultado.ejecutar`
        self._grupo = grupo

    async def notificar(self, trabajo: Trabajo) -> None:
        if trabajo.url_notificacion is None:
            return
        self._grupo.start_soon(self._entregar_protegido, trabajo)

    async def _entregar_protegido(self, trabajo: Trabajo) -> None:
        try:
            await self._entregar(trabajo)
        except Exception:
            logger.exception(
                "notificacion_error",
                extra={"id_evento": str(trabajo.id_evento), "id_trabajo": str(trabajo.id)},
            )
