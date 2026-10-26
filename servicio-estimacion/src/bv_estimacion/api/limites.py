"""Tope del tamaño del cuerpo de los `POST`, **antes** de que nadie lo parsee.

Hay dos defensas, y las dos hacen falta:

1. Si el pedido declara `Content-Length` y se pasa del tope, se rechaza sin leer el cuerpo.
2. Si no lo declara (`Transfer-Encoding: chunked`), el tope se aplica **mientras se lee**: se cuentan los bytes que
   llegan y se corta en cuanto se pasa. Un encabezado solo no alcanza: el cliente puede no mandarlo.

El tope de un multipart es `n × MAX_MB_IMAGEN` más un margen para los demás campos y los encabezados de cada parte;
el de un JSON es chico (solo lleva URLs y metadatos). Cada archivo, además, se valida por su cuenta contra
`MAX_MB_IMAGEN`.
"""

from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from bv_estimacion.api.errores import problema
from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio
from bv_estimacion.dominio.estimacion import MAX_IMAGENES

MARGEN_MULTIPART_BYTES = 1024 * 1024
MAX_JSON_BYTES = 64 * 1024  # propuesto

# Cuántas imágenes admite cada ruta: de ahí sale el tope de su multipart.
IMAGENES_POR_RUTA = {"/v1/estimaciones": MAX_IMAGENES, "/v1/validaciones": 1}


class LimitarCuerpo:
    def __init__(self, app: ASGIApp, *, max_bytes_imagen: int) -> None:
        self.app = app
        self._max_bytes_imagen = max_bytes_imagen

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in IMAGENES_POR_RUTA:
            await self.app(scope, receive, send)
            return

        encabezados = Headers(scope=scope)
        tipo = encabezados.get("content-type", "").split(";")[0].strip().lower()
        if tipo == "multipart/form-data":
            limite = IMAGENES_POR_RUTA[scope["path"]] * self._max_bytes_imagen + MARGEN_MULTIPART_BYTES
            codigo, detalle = CodigoError.IMAGEN_MUY_GRANDE, "El pedido supera el tamaño máximo permitido."
        else:
            limite = MAX_JSON_BYTES
            codigo, detalle = CodigoError.IMAGEN_MUY_GRANDE, "El cuerpo del pedido supera los 64 KB."

        declarado = encabezados.get("content-length", "")
        if declarado.isdigit() and int(declarado) > limite:
            await problema(Request(scope, receive), codigo, detalle)(scope, receive, send)
            return

        recibido = 0

        async def recibir() -> Message:
            nonlocal recibido
            mensaje = await receive()
            if mensaje["type"] == "http.request":
                recibido += len(mensaje.get("body", b""))
                if recibido > limite:
                    # Sube hasta el manejador de errores del endpoint que esté leyendo el cuerpo.
                    raise ErrorDeDominio(codigo, detalle)
            return mensaje

        await self.app(scope, recibir, send)
