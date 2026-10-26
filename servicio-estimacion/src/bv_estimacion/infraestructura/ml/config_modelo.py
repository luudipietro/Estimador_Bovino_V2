"""Versión y rango de validez del modelo, leídos de `config.json`. El rango nunca se escribe a mano (R-20, R-23)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bv_estimacion.dominio.valores import VersionModelo


class ConfigModeloInvalida(Exception):
    """El `config.json` del modelo falta o no tiene lo necesario."""


@dataclass(frozen=True, slots=True)
class FichaModelo:
    """Lo que `GET /v1/modelo` cuenta del modelo. Lo opcional es `None` si el `config.json` no lo trae: no se
    inventa ni se saca del texto libre de `version`."""

    version: VersionModelo
    mape_test: float | None = None
    ic95_test: tuple[float, float] | None = None
    fecha_entrenamiento: str | None = None


def _numero_opcional(datos: dict[str, Any], clave: str) -> float | None:
    valor = datos.get(clave)
    if valor is None:
        return None
    if isinstance(valor, bool) or not isinstance(valor, int | float):
        raise ValueError(f"{clave} tiene que ser un número")
    return float(valor)


def _intervalo_opcional(datos: dict[str, Any], clave: str) -> tuple[float, float] | None:
    valor = datos.get(clave)
    if valor is None:
        return None
    if not isinstance(valor, list) or len(valor) != 2:
        raise ValueError(f"{clave} tiene que ser una lista de dos números")
    minimo, maximo = valor
    return (float(minimo), float(maximo))


def _texto_opcional(datos: dict[str, Any], clave: str) -> str | None:
    valor = datos.get(clave)
    if valor is None:
        return None
    if not isinstance(valor, str) or not valor.strip():
        raise ValueError(f"{clave} tiene que ser un texto no vacío")
    return valor


def cargar_ficha_modelo(carpeta_modelos: Path) -> FichaModelo:
    ruta = carpeta_modelos / "config.json"
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        minimo, maximo = datos["rango_entrenamiento_kg"]
        return FichaModelo(
            version=VersionModelo(str(datos["version"]), float(minimo), float(maximo)),
            mape_test=_numero_opcional(datos, "mape_test"),
            ic95_test=_intervalo_opcional(datos, "ic95_test"),
            fecha_entrenamiento=_texto_opcional(datos, "fecha_entrenamiento"),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        # ValueError cubre el JSON inválido y también un rango o versión que el dominio rechaza
        raise ConfigModeloInvalida(f"no se pudo leer la versión del modelo de {ruta}") from error


def cargar_version_modelo(carpeta_modelos: Path) -> VersionModelo:
    return cargar_ficha_modelo(carpeta_modelos).version
