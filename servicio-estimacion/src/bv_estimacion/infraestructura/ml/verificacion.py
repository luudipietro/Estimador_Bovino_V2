"""Verificación de los pesos del modelo contra `SHA256SUMS.txt`. Si algo no coincide, la app no arranca."""

import hashlib
import json
from pathlib import Path

_BLOQUE = 1024 * 1024
_CLAVES_DE_ARCHIVO = ("archivo_pose", "archivo_cnn", "archivo_detector")


class VerificacionModeloFallida(Exception):
    """Falta un archivo del modelo o su hash no coincide con el declarado."""


def sha256_de_archivo(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(_BLOQUE), b""):
            resumen.update(bloque)
    return resumen.hexdigest()


def leer_sumas(ruta: Path) -> dict[str, str]:
    """Lee `SHA256SUMS.txt` (formato de `sha256sum`: `<hash> *<archivo>`). Devuelve {archivo: hash}."""
    try:
        lineas = ruta.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise VerificacionModeloFallida(f"no se pudo leer {ruta.name}") from error

    sumas: dict[str, str] = {}
    for linea in lineas:
        if not linea.strip():
            continue
        partes = linea.split(maxsplit=1)
        if len(partes) != 2 or len(partes[0]) != 64:
            raise VerificacionModeloFallida(f"línea mal formada en {ruta.name}")
        sumas[partes[1].strip().lstrip("*")] = partes[0].lower()
    if not sumas:
        raise VerificacionModeloFallida(f"{ruta.name} está vacío")
    return sumas


def verificar_hashes_modelos(carpeta_modelos: Path) -> None:
    """Comprueba que cada archivo listado exista y coincida, y que **todos los que usa `config.json`** estén
    cubiertos por la lista: un `.pt` fuera de ella podría reemplazarse sin que nadie lo note."""
    sumas = leer_sumas(carpeta_modelos / "SHA256SUMS.txt")

    for nombre, esperado in sumas.items():
        ruta = carpeta_modelos / nombre
        if not ruta.is_file():
            raise VerificacionModeloFallida(f"falta el archivo del modelo {nombre}")
        if sha256_de_archivo(ruta) != esperado:
            raise VerificacionModeloFallida(f"el archivo {nombre} no coincide con su hash")

    try:
        config = json.loads((carpeta_modelos / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise VerificacionModeloFallida("no se pudo leer config.json") from error
    for clave in _CLAVES_DE_ARCHIVO:
        archivo = config.get(clave)
        if archivo and archivo not in sumas:
            raise VerificacionModeloFallida(f"{archivo} (de config.json) no figura en SHA256SUMS.txt")
