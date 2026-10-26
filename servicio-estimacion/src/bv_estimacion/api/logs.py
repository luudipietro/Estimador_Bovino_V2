"""Logs JSON, una línea por evento. Campos fijos: `ts`, `nivel`, `evento`, `request_id`, `id_evento`, `id_trabajo` y
`version_modelo`; lo demás que llegue en `extra` se agrega tal cual.

**Nunca** imágenes, URLs firmadas completas, tokens ni coordenadas GPS: todo texto pasa por `sanear`, que reemplaza
las URLs y los tokens `Bearer` aunque vengan dentro del mensaje de una excepción.
"""

import json
import logging
import re
import sys
import traceback
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class ContextoDeLogs:
    """Lo que es igual para todo el proceso. El `lifespan` fija la versión cuando el modelo ya está cargado. Es estado
    de configuración del logging (como los manejadores del root logger), no estado de negocio."""

    version_modelo: str | None = None


CONTEXTO = ContextoDeLogs()

_URL = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
_BEARER = re.compile(r"(bearer\s+)\S+", re.IGNORECASE)
_ESTANDAR = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "taskName"}


def sanear(texto: str) -> str:
    return _BEARER.sub(r"\1<oculto>", _URL.sub("<url>", texto))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        registro: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "nivel": record.levelname,
            "evento": sanear(record.getMessage()),
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
            "id_evento": getattr(record, "id_evento", None),
            "id_trabajo": getattr(record, "id_trabajo", None),
            "version_modelo": getattr(record, "version_modelo", None) or CONTEXTO.version_modelo,
        }
        for nombre, valor in vars(record).items():
            if nombre not in _ESTANDAR and nombre not in registro:
                registro[nombre] = sanear(valor) if isinstance(valor, str) else valor
        if record.exc_info and record.exc_info[0] is not None:
            registro["error"] = record.exc_info[0].__name__  # la clase sirve para diagnosticar
            registro["traza"] = sanear("".join(traceback.format_exception(*record.exc_info)))
        return json.dumps(registro, ensure_ascii=False, default=str)


def configurar_logs(nivel: int = logging.INFO) -> None:
    """Deja un único manejador JSON en el root logger. Se puede llamar varias veces."""
    raiz = logging.getLogger()
    raiz.setLevel(nivel)
    if any(isinstance(h.formatter, JsonFormatter) for h in raiz.handlers):
        return
    manejador = logging.StreamHandler(sys.stdout)
    manejador.setFormatter(JsonFormatter())
    raiz.addHandler(manejador)
