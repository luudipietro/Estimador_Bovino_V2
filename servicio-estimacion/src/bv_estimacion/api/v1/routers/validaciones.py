import anyio
from fastapi import APIRouter, Depends, Request

from bv_estimacion.api.dependencias import Contenedor, obtener_contenedor
from bv_estimacion.api.seguridad import exigir_token
from bv_estimacion.api.v1.entrada import leer_imagen_de_validacion
from bv_estimacion.api.v1.esquemas.pedido import ResultadoValidacionRespuesta
from bv_estimacion.aplicacion.puertos import ModeloNoDisponible
from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio

router = APIRouter(prefix="/v1/validaciones", tags=["validaciones"], dependencies=[Depends(exigir_token)])


@router.post(
    "",
    response_model=ResultadoValidacionRespuesta,
    summary="Validación rápida de calidad, sin estimar el peso",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["imagen"],
                        "properties": {"imagen": {"type": "string", "format": "binary"}},
                        "additionalProperties": False,
                    }
                }
            },
        }
    },
)
async def validar(
    request: Request,
    contenedor: Contenedor = Depends(obtener_contenedor),  # noqa: B008
) -> ResultadoValidacionRespuesta:
    """Una imagen en el campo `imagen`. Un formato, resolución o archivo inválido es un error 4xx con su código; una
    foto válida pero de mala calidad (sin animal, tapada, cortada, borrosa) es `apta: false` con sus motivos. No crea
    ningún trabajo, así que no pide `Idempotency-Key`."""
    config = contenedor.configuracion
    datos = await leer_imagen_de_validacion(request, config)
    try:
        with anyio.fail_after(config.presupuesto_estimacion_s):
            resultado = await contenedor.validar.ejecutar(datos)
    except TimeoutError as error:
        raise ErrorDeDominio(CodigoError.TIEMPO_EXCEDIDO, "La validación superó el presupuesto de tiempo.") from error
    except ModeloNoDisponible as error:
        raise ErrorDeDominio(CodigoError.MODELO_NO_DISPONIBLE, "El modelo no está disponible.") from error
    return ResultadoValidacionRespuesta(
        apta=resultado.apta, motivos=list(resultado.motivos), advertencias=list(resultado.advertencias)
    )
