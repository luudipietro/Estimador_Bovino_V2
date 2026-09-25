"""Lectura del dataset AcmeAI/BMGF: nombres de archivo, keypoints COCO y rutas.

Particularidades verificadas sobre los datos crudos (no asumir lo contrario):
- Vista lateral con 9 keypoints con nombre: solo lotes B3 y B4. B2 usa otros esquemas (6 y 23 kp) y tiene
  nombres irregulares -> fuera de la v1.
- El ORDEN de los keypoints dentro de cada anotación difiere entre B3 y B4 -> mapear siempre por nombre.
- El "id" del nombre de archivo NO identifica con certeza un animal único (mismo id con varios pesos).
  animal_id = id numérico, compartido entre lotes, para que un split por animal sea conservador.
- Resolución distinta por lote: B3 4160x3120, B4 1900x1425 -> los píxeles no son comparables entre lotes.
"""
import re
from pathlib import Path

# Nombre canónico de cada keypoint del esquema de 9 puntos (mismo significado en B3 y B4)
KP_CANONICOS = [
    "wither", "pinbone", "shoulderbone",
    "front_girth_top", "front_girth_bottom",
    "rear_girth_top", "rear_girth_bottom",
    "height_top", "height_bottom",
]

_RX = {
    "B3": re.compile(r"^(?P<id>\d+)_s_(?P<peso>\d+)_(?P<sexo>[MF])\.jpg$"),
    "B4": re.compile(r"^(?P<id>\d+)_b4-(?P<sub>\d+)_s_(?P<peso>\d+)_(?P<sexo>[MF])\.jpg$"),
}

# Carpeta de imágenes/máscaras y JSON de keypoints (relativos a la raíz del dataset)
LOTES_SIDE = {
    "B3": {"imagenes": "Pixel/B3/images", "mascaras": "Pixel/B3/annotations",
           "coco": "Vector/B3/Side/data/COCO_Side.json"},
    "B4": {"imagenes": "Pixel/B4/Side/images", "mascaras": "Pixel/B4/Side/annotations",
           "coco": "Vector/B4/Side/data/coco_b4_side.json"},
}


def parsear_nombre(lote: str, nombre: str) -> dict | None:
    """Extrae id/peso/sexo del nombre de archivo. Devuelve None si no respeta la convención del lote."""
    m = _RX[lote].match(nombre)
    if not m:
        return None
    g = m.groupdict()
    return {
        "animal_id": g["id"],
        "subbatch": g.get("sub"),
        "peso_kg": float(g["peso"]),
        "sexo": g["sexo"],
    }


def nombre_kp_canonico(nombre_json: str) -> str:
    """'4_front_girth_top' / '8_Height_top' -> 'front_girth_top' / 'height_top' (quita el prefijo numérico)."""
    return re.sub(r"^\d+_", "", nombre_json).lower()


def kp_por_nombre(keypoints_flat: list, nombres_json: list[str]) -> dict[str, tuple[float, float, int]]:
    """Lista plana COCO [x1,y1,v1,x2,...] + nombres en el orden del JSON -> {nombre_canónico: (x, y, v)}."""
    assert len(keypoints_flat) == 3 * len(nombres_json), "keypoints y nombres no coinciden en cantidad"
    out = {}
    for i, n in enumerate(nombres_json):
        x, y, v = keypoints_flat[3 * i: 3 * i + 3]
        out[nombre_kp_canonico(n)] = (float(x), float(y), int(v))
    return out


def ruta_mascara(dir_mascaras: Path, nombre_imagen: str) -> Path:
    return dir_mascaras / f"{nombre_imagen}___fuse.png"
