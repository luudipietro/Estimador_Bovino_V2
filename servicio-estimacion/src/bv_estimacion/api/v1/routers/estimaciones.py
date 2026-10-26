from typing import Any

from fastapi import APIRouter, Depends, Header, Request, Response
from starlette.responses import JSONResponse

from bv_estimacion.api.dependencias import Contenedor, obtener_contenedor
from bv_estimacion.api.seguridad import exigir_token
from bv_estimacion.api.v1.entrada import leer_id_evento, leer_pedido_de_estimacion
from bv_estimacion.api.v1.esquemas.pedido import TrabajoAceptado
from bv_estimacion.api.v1.esquemas.trabajo import RespuestaTrabajo, serializar_trabajo
from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio
from bv_estimacion.dominio.valores import IdTrabajo

router = APIRouter(prefix="/v1/estimaciones", tags=["estimaciones"], dependencies=[Depends(exigir_token)])

_CUERPO_DEL_PEDIDO: dict[str, Any] = {
    "required": True,
    "description": (
        "Dos formas, mutuamente excluyentes: **multipart/form-data** con 1 a 4 archivos en `imagenes`, o "
        "**application/json** con 1 a 4 `referencias_imagen` (URL firmada y SHA-256). Máximo 64 KB de JSON."
    ),
    "content": {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "required": ["imagenes"],
                "properties": {
                    "imagenes": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {"type": "string", "format": "binary"},
                    },
                    "metadatos": {"type": "string", "description": "Un objeto JSON. Vuelve sin modificar (R-26)."},
                    "url_notificacion": {"type": "string", "format": "uri", "description": "Host en la allowlist"},
                },
                "additionalProperties": False,
            }
        },
        "application/json": {
            "schema": {
                "type": "object",
                "required": ["referencias_imagen"],
                "properties": {
                    "referencias_imagen": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "required": ["url", "sha256"],
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "format": "uri",
                                    "description": "https, host en la allowlist",
                                },
                                "sha256": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$"},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "metadatos": {"type": "object", "description": "Vuelve sin modificar (R-26)."},
                    "url_notificacion": {"type": "string", "format": "uri", "description": "Host en la allowlist"},
                },
                "additionalProperties": False,
            }
        },
    },
}


@router.post(
    "",
    status_code=202,
    response_model=TrabajoAceptado,
    summary="Solicita una estimación",
    responses={200: {"model": TrabajoAceptado, "description": "El `id_evento` ya existía con las mismas imágenes"}},
    openapi_extra={"requestBody": _CUERPO_DEL_PEDIDO},
)
async def crear(
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", description="El `id_evento`"),  # noqa: B008
    contenedor: Contenedor = Depends(obtener_contenedor),  # noqa: B008
) -> JSONResponse:
    """Valida el pedido en el borde y lo deja en cola. **202** para un trabajo nuevo; **200** si el `id_evento` ya
    existía con las mismas imágenes (en cualquier orden), con el estado actual del trabajo existente."""
    id_evento = leer_id_evento(idempotency_key)
    comando = await leer_pedido_de_estimacion(request, contenedor.configuracion, id_evento)
    aceptada = await contenedor.solicitar.ejecutar(comando)
    trabajo = aceptada.trabajo
    cuerpo = TrabajoAceptado(id=str(trabajo.id), id_evento=str(trabajo.id_evento), estado=trabajo.estado.value)  # type: ignore[arg-type]  # el mismo conjunto de valores
    return JSONResponse(
        cuerpo.model_dump(),
        status_code=202 if aceptada.nuevo else 200,
        headers={"Location": f"/v1/estimaciones/{trabajo.id}"},
    )


@router.get(
    "/{id_trabajo}",
    response_model=RespuestaTrabajo,
    summary="Estado y resultado de una estimación",
)
async def consultar(
    id_trabajo: str,
    contenedor: Contenedor = Depends(obtener_contenedor),  # noqa: B008
) -> Response:
    """Devuelve **exactamente** los mismos bytes que se envían por callback."""
    if not id_trabajo.strip():
        raise ErrorDeDominio(CodigoError.TRABAJO_NO_ENCONTRADO, "El trabajo no existe o ya venció.")
    trabajo = await contenedor.consultar.ejecutar(IdTrabajo(id_trabajo))
    return Response(serializar_trabajo(trabajo), media_type="application/json")
