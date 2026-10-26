"""El `openapi.json` que lee el equipo del adaptador C#, con lo que FastAPI no puede deducir solo: la autenticación
Bearer, el formato `problem+json` de los errores y los estados HTTP de cada operación."""

import copy
import json
from typing import Any

from fastapi import FastAPI

from bv_estimacion.dominio.errores import CodigoError

REF_PROBLEMA = {"$ref": "#/components/schemas/Problema"}

SCHEMA_PROBLEMA: dict[str, Any] = {
    "type": "object",
    "description": "Error en formato RFC 9457 (`application/problem+json`). Los códigos están en `docs/errores.md`.",
    "required": ["type", "title", "status", "detail", "codigo", "request_id"],
    "properties": {
        "type": {"type": "string", "description": "Identificador del error; no tiene por qué resolver"},
        "title": {"type": "string"},
        "status": {"type": "integer"},
        "detail": {"type": "string", "description": "Texto legible. Nunca trazas ni nombres de clases"},
        "codigo": {"type": "string", "enum": sorted(c.value for c in CodigoError)},
        "regla": {"type": "string", "description": "Regla de dominio involucrada (por ejemplo `R-18`), si aplica"},
        "campos": {
            "type": "array",
            "items": {"type": "string"},
            "description": "En `solicitud_invalida`: los nombres de lo que está mal, nunca los valores recibidos",
        },
        "request_id": {"type": ["string", "null"], "description": "El mismo valor de `X-Request-ID`"},
    },
}

DESCRIPCION_ERROR = {
    "400": "Pedido mal formado (`solicitud_invalida`)",
    "401": "Falta el token de servicio o no es válido (`no_autenticado`)",
    "404": "El trabajo no existe o ya venció (`trabajo_no_encontrado`)",
    "409": "El `id_evento` ya se usó con otras imágenes (`id_evento_en_conflicto`)",
    "413": "La imagen o el pedido supera el tamaño máximo (`imagen_muy_grande`)",
    "415": "Formato de imagen no admitido (`formato_no_admitido`)",
    "422": (
        "Entrada inválida: `imagen_corrupta`, `resolucion_insuficiente`, `cantidad_imagenes_invalida` o "
        "`referencia_no_permitida`"
    ),
    "500": "Error interno (`error_interno`)",
    "503": "`cola_llena` (con `Retry-After`) o `modelo_no_disponible`",
    "504": "La validación superó el presupuesto de tiempo (`tiempo_excedido`)",
}

# Qué errores puede devolver cada operación, además del 401 y el 500 que tienen todas las rutas autenticadas.
ERRORES_POR_OPERACION: dict[tuple[str, str], tuple[str, ...]] = {
    ("/v1/estimaciones", "post"): ("400", "409", "413", "415", "422", "503"),
    ("/v1/estimaciones/{id_trabajo}", "get"): ("404",),
    ("/v1/validaciones", "post"): ("400", "413", "415", "422", "503", "504"),
}


def _respuesta_de_error(estado: str) -> dict[str, Any]:
    return {
        "description": DESCRIPCION_ERROR[estado],
        "content": {"application/problem+json": {"schema": REF_PROBLEMA}},
    }


def _personalizar(esquema: dict[str, Any]) -> None:
    componentes = esquema.setdefault("components", {})
    schemas = componentes.setdefault("schemas", {})
    # FastAPI documenta `422 HTTPValidationError`, pero este servicio devuelve `400 solicitud_invalida` en problem+json.
    schemas.pop("HTTPValidationError", None)
    schemas.pop("ValidationError", None)
    schemas["Problema"] = SCHEMA_PROBLEMA
    componentes["securitySchemes"] = {"BearerAuth": {"type": "http", "scheme": "bearer"}}

    for ruta, operaciones in esquema["paths"].items():
        for metodo, operacion in operaciones.items():
            respuestas = operacion.setdefault("responses", {})
            respuestas.pop("422", None)  # el que agrega FastAPI; el 422 real de cada ruta se documenta abajo
            if ruta == "/salud":
                continue  # la única ruta abierta
            operacion["security"] = [{"BearerAuth": []}]
            for estado in ("401", *ERRORES_POR_OPERACION.get((ruta, metodo), ()), "500"):
                respuestas[estado] = _respuesta_de_error(estado)


def esquema_openapi() -> dict[str, Any]:
    """Genera el esquema desde la app real, sin arrancarla (no se carga el modelo)."""
    from bv_estimacion.api.main import create_app
    from bv_estimacion.infraestructura.config import Configuracion

    configuracion = Configuracion(
        token_servicio="solo-para-el-esquema",
        secreto_hmac="solo-para-el-esquema",
        _env_file=None,  # type: ignore[call-arg]  # es un argumento propio de pydantic-settings
    )
    app: FastAPI = create_app(configuracion, configurar_logs=False)
    esquema = copy.deepcopy(app.openapi())
    _personalizar(esquema)
    return esquema


def serializar_esquema(esquema: dict[str, Any]) -> str:
    """Siempre igual para el mismo esquema (claves ordenadas, UTF-8, terminado en salto de línea)."""
    return json.dumps(esquema, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
