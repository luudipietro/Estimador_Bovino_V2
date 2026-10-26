"""Versión y rango de validez del modelo, leídos de `config.json`. El rango nunca se escribe a mano (R-20, R-23)."""

import json
from pathlib import Path

from bv_estimacion.dominio.valores import VersionModelo


class ConfigModeloInvalida(Exception):
    """El `config.json` del modelo falta o no tiene lo necesario."""


def cargar_version_modelo(carpeta_modelos: Path) -> VersionModelo:
    ruta = carpeta_modelos / "config.json"
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        minimo, maximo = datos["rango_entrenamiento_kg"]
        return VersionModelo(str(datos["version"]), float(minimo), float(maximo))
    except (OSError, ValueError, KeyError, TypeError) as error:
        # ValueError cubre el JSON inválido y también un rango o versión que el dominio rechaza
        raise ConfigModeloInvalida(f"no se pudo leer la versión del modelo de {ruta}") from error
