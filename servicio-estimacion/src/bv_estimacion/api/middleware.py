"""Middleware ASGI puro (no `BaseHTTPMiddleware`: no parte el cuerpo ni rompe el streaming)."""

import logging
import re
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from bv_estimacion.api.logs import request_id_var

logger = logging.getLogger(__name__)

ENCABEZADO_REQUEST_ID = "X-Request-ID"
CLAVE_EN_SCOPE = "bv_request_id"
_ID_VALIDO = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def request_id_de(scope: Scope) -> str | None:
    valor = scope.get(CLAVE_EN_SCOPE)
    return valor if isinstance(valor, str) else None


class ContextoDePeticion:
    """Le da a cada petición un `request_id` (el que mande el cliente si es razonable, o uno nuevo), lo pone en el
    contexto de los logs y en la respuesta, y deja una línea de acceso. De la petición solo registra método, ruta,
    estado y duración: ni la query string ni los encabezados, que pueden llevar el token o una URL firmada."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        propuesto = Headers(scope=scope).get(ENCABEZADO_REQUEST_ID, "")
        request_id = propuesto if _ID_VALIDO.fullmatch(propuesto) else uuid.uuid4().hex
        scope[CLAVE_EN_SCOPE] = request_id  # los manejadores de errores 500 corren fuera de este contexto
        token = request_id_var.set(request_id)
        inicio = time.perf_counter()
        estado = 500  # si la excepción se escapa, la respuesta la arma ServerErrorMiddleware

        async def enviar(mensaje: Message) -> None:
            nonlocal estado
            if mensaje["type"] == "http.response.start":
                estado = mensaje["status"]
                MutableHeaders(scope=mensaje)[ENCABEZADO_REQUEST_ID] = request_id
            await send(mensaje)

        try:
            await self.app(scope, receive, enviar)
        finally:
            logger.info(
                "peticion",
                extra={
                    "request_id": request_id,
                    "metodo": scope["method"],
                    "ruta": scope["path"],
                    "estado": estado,
                    "duracion_ms": round((time.perf_counter() - inicio) * 1000),
                },
            )
            request_id_var.reset(token)
