"""Fixtures compartidas. Los pesos reales (`.pt`) no se versionan: las pruebas que los necesitan llevan la marca
`modelo` y se saltean (skip) si los archivos no están, para que la CI y los compañeros sin pesos no fallen."""

from pathlib import Path

import pytest

RAIZ_REPO = Path(__file__).resolve().parents[2]
CARPETA_MODELOS_REPO = RAIZ_REPO / "estimador" / "modelos"


def faltantes_de_los_pesos(carpeta: Path) -> list[str]:
    """Nombres de los archivos listados en `SHA256SUMS.txt` que no están en `carpeta`."""
    lista = carpeta / "SHA256SUMS.txt"
    if not lista.is_file():
        return ["SHA256SUMS.txt"]
    nombres = [
        linea.split(maxsplit=1)[1].strip().lstrip("*")
        for linea in lista.read_text("utf-8").splitlines()
        if linea.strip()
    ]
    return [nombre for nombre in nombres if not (carpeta / nombre).is_file()]


@pytest.fixture(scope="session")
def carpeta_modelos_reales() -> Path:
    """La carpeta con los `.pt` reales, o skip si no se descargaron (`python ../estimador/descargar_modelos.py`)."""
    faltan = faltantes_de_los_pesos(CARPETA_MODELOS_REPO)
    if faltan:
        pytest.skip(f"faltan los pesos del modelo ({', '.join(faltan)}); correr descargar_modelos.py")
    return CARPETA_MODELOS_REPO
