"""Errores como `application/problem+json` (RFC 9457): `type`, `title`, `status`, `detail`, `codigo`, `regla` (si
aplica) y `request_id`. Los códigos son los de `docs/errores.md`.

Nunca se exponen trazas, rutas ni nombres de clases: el detalle de una falla interna va al log, con el `request_id`.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from bv_estimacion.api.middleware import ENCABEZADO_REQUEST_ID, request_id_de
from bv_estimacion.api.v1.esquemas.trabajo import DETALLE_DE_FALLA
from bv_estimacion.dominio.errores import FALLAS_INTERNAS, CodigoError, ErrorDeDominio

logger = logging.getLogger(__name__)

TIPO_BASE = "https://bovinovision/errores/"  # identificador, no tiene por qué resolver
RETRY_AFTER_COLA_LLENA_S = 5  # propuesto, a confirmar junto con `cola_llena`

ESTADO_HTTP: dict[CodigoError, int] = {
    CodigoError.FORMATO_NO_ADMITIDO: 415,
    CodigoError.IMAGEN_CORRUPTA: 422,
    CodigoError.IMAGEN_MUY_GRANDE: 413,
    CodigoError.RESOLUCION_INSUFICIENTE: 422,
    CodigoError.CANTIDAD_IMAGENES_INVALIDA: 422,
    CodigoError.ID_EVENTO_EN_CONFLICTO: 409,
    CodigoError.REFERENCIA_NO_PERMITIDA: 422,
    CodigoError.NO_AUTENTICADO: 401,
    CodigoError.TRABAJO_NO_ENCONTRADO: 404,
    CodigoError.COLA_LLENA: 503,
    CodigoError.SOLICITUD_INVALIDA: 400,
    # Las fallas internas dejan un trabajo `fallido`; solo llegan a una respuesta HTTP si un endpoint síncrono
    # (como `/v1/validaciones`) las sufre.
    CodigoError.MODELO_NO_DISPONIBLE: 503,
    CodigoError.TIEMPO_EXCEDIDO: 504,
    CodigoError.REFERENCIA_INACCESIBLE: 502,
    CodigoError.ERROR_INTERNO: 500,
}

TITULO: dict[CodigoError, str] = {
    CodigoError.FORMATO_NO_ADMITIDO: "Formato de imagen no admitido",
    CodigoError.IMAGEN_CORRUPTA: "Imagen corrupta",
    CodigoError.IMAGEN_MUY_GRANDE: "Imagen demasiado grande",
    CodigoError.RESOLUCION_INSUFICIENTE: "Resolución insuficiente",
    CodigoError.CANTIDAD_IMAGENES_INVALIDA: "Cantidad de imágenes inválida",
    CodigoError.ID_EVENTO_EN_CONFLICTO: "El id_evento ya se usó con otras imágenes",
    CodigoError.REFERENCIA_NO_PERMITIDA: "Referencia no permitida",
    CodigoError.NO_AUTENTICADO: "No autenticado",
    CodigoError.TRABAJO_NO_ENCONTRADO: "Trabajo no encontrado",
    CodigoError.COLA_LLENA: "Cola de trabajos llena",
    CodigoError.SOLICITUD_INVALIDA: "Solicitud inválida",
    CodigoError.MODELO_NO_DISPONIBLE: "Modelo no disponible",
    CodigoError.TIEMPO_EXCEDIDO: "Tiempo excedido",
    CodigoError.REFERENCIA_INACCESIBLE: "Referencia inaccesible",
    CodigoError.ERROR_INTERNO: "Error interno",
}


class SolicitudInvalida(ErrorDeDominio):
    """Un pedido mal formado (400). `campos` son los **nombres** de lo que está mal, nunca los valores recibidos."""

    def __init__(self, detalle: str, campos: list[str] | None = None) -> None:
        super().__init__(CodigoError.SOLICITUD_INVALIDA, detalle)
        self.campos = [campo[:64] for campo in (campos or [])]  # un nombre de campo lo puso el cliente: se acota


_DETALLE_HTTP = {404: "La ruta no existe.", 405: "Método no permitido para esta ruta."}


def problema(
    request: Request,
    codigo: CodigoError,
    detalle: str,
    *,
    estado: int | None = None,
    regla: str | None = None,
    cabeceras: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    request_id = request_id_de(request.scope)
    estado = estado or ESTADO_HTTP[codigo]
    cuerpo: dict[str, Any] = {
        "type": f"{TIPO_BASE}{codigo.value}",
        "title": TITULO[codigo],
        "status": estado,
        "detail": detalle,
        "codigo": codigo.value,
    }
    if regla is not None:
        cuerpo["regla"] = regla
    cuerpo |= extra or {}
    cuerpo["request_id"] = request_id
    # Este manejador puede correr fuera del middleware que agrega el encabezado (ServerErrorMiddleware): se pone acá.
    encabezados = {ENCABEZADO_REQUEST_ID: request_id or "", **(cabeceras or {})}
    return JSONResponse(cuerpo, status_code=estado, media_type="application/problem+json", headers=encabezados)


async def _error_de_dominio(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, ErrorDeDominio)
    cabeceras: dict[str, str] = {}
    if error.codigo is CodigoError.NO_AUTENTICADO:
        cabeceras["WWW-Authenticate"] = "Bearer"
    if error.codigo is CodigoError.COLA_LLENA:
        cabeceras["Retry-After"] = str(RETRY_AFTER_COLA_LLENA_S)

    if error.codigo in FALLAS_INTERNAS:
        # El mensaje con que se lanzó es para el log, no para el cliente.
        logger.error("error_de_dominio", exc_info=error, extra={"codigo": error.codigo.value})
        detalle = DETALLE_DE_FALLA[error.codigo]
    else:
        detalle = error.detalle
    extra = {"campos": error.campos} if isinstance(error, SolicitudInvalida) and error.campos else None
    return problema(request, error.codigo, detalle, regla=error.regla, cabeceras=cabeceras, extra=extra)


async def _validacion(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, RequestValidationError)
    # Solo los nombres de los campos: el error de pydantic trae el valor recibido, que puede ser cualquier cosa.
    campos = sorted({".".join(str(parte) for parte in e["loc"][1:] or e["loc"]) for e in error.errors()})
    return problema(
        request,
        CodigoError.SOLICITUD_INVALIDA,
        "El pedido está mal formado.",
        extra={"campos": campos},
    )


async def _http(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, StarletteHTTPException)
    detalle = _DETALLE_HTTP.get(error.status_code, "La solicitud no es válida.")
    cabeceras = dict(error.headers) if error.headers else None
    return problema(request, CodigoError.SOLICITUD_INVALIDA, detalle, estado=error.status_code, cabeceras=cabeceras)


async def _no_controlado(request: Request, error: Exception) -> JSONResponse:
    logger.error(
        "error_no_controlado",
        exc_info=error,
        extra={"request_id": request_id_de(request.scope), "ruta": request.url.path},
    )
    return problema(request, CodigoError.ERROR_INTERNO, DETALLE_DE_FALLA[CodigoError.ERROR_INTERNO])


def registrar_manejadores(app: FastAPI) -> None:
    app.add_exception_handler(ErrorDeDominio, _error_de_dominio)
    app.add_exception_handler(RequestValidationError, _validacion)
    app.add_exception_handler(StarletteHTTPException, _http)
    app.add_exception_handler(Exception, _no_controlado)
