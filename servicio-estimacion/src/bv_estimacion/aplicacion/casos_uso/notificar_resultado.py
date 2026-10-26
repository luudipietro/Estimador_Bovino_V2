"""RF-21 (soporte): avisar al backend que un trabajo terminó. La notificación al usuario la emite el backend."""

import logging

from bv_estimacion.aplicacion.puertos import Dormidor, EnviadorCallback
from bv_estimacion.dominio.estimacion import Trabajo

logger = logging.getLogger(__name__)


class NotificarResultado:
    """Entrega el callback con reintentos y backoff exponencial.

    Corre en una tarea aparte, **después** de guardar el estado final: no ocupa el lugar del modelo, no retrasa la
    cola y no cuenta dentro del presupuesto de tiempo. Si todos los intentos fallan **no cambia el estado del
    trabajo**: el resultado sigue disponible por `GET` y el backend nunca depende solo del callback.
    """

    def __init__(self, enviador: EnviadorCallback, dormidor: Dormidor, *, intentos: int, espera_base_s: float) -> None:
        if intentos < 1:
            raise ValueError("hace falta al menos un intento")
        self._enviador = enviador
        self._dormidor = dormidor
        self._intentos = intentos
        self._espera_base_s = espera_base_s

    async def ejecutar(self, trabajo: Trabajo) -> bool:
        """`True` si el backend lo recibió. Nunca lanza: es una tarea de fondo."""
        if trabajo.url_notificacion is None:
            return False

        for intento in range(1, self._intentos + 1):
            contexto = {"id_evento": str(trabajo.id_evento), "id_trabajo": str(trabajo.id), "intento": intento}
            try:
                await self._enviador.enviar(trabajo.url_notificacion, trabajo)
            except Exception:  # un fallo del envío, cualquiera sea, no puede escapar de la tarea de fondo
                logger.warning("callback_fallido", extra=contexto)
                if intento < self._intentos:
                    await self._dormidor.dormir(self._espera_base_s * 2 ** (intento - 1))
            else:
                logger.info("callback_entregado", extra=contexto)
                return True

        logger.error(
            "callback_agotado",
            extra={"id_evento": str(trabajo.id_evento), "id_trabajo": str(trabajo.id), "intento": self._intentos},
        )
        return False
