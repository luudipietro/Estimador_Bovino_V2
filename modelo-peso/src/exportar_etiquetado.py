"""Exporta las imagenes laterales listas para etiquetar en CVAT o Roboflow.

Las reduce a 1024 px de ancho: alcanza de sobra para marcar landmarks y hace
que la herramienta de etiquetado vaya fluida (las originales son 4032x3024).

Uso:  python src/exportar_etiquetado.py
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

ANCHO = 1024
SALIDA = Path(__file__).resolve().parents[1] / "etiquetado" / "imagenes"


def guardar(ruta: Path, img, calidad=92):
    """cv2.imwrite falla en silencio con rutas no-ASCII en Windows, y este
    proyecto vive en una carpeta con 'año'."""
    ok, buf = cv2.imencode(ruta.suffix, img, [cv2.IMWRITE_JPEG_QUALITY, calidad])
    if not ok:
        raise IOError(f"no se pudo codificar {ruta}")
    ruta.write_bytes(buf.tobytes())


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    origen = [p for p in config.CRUDO.rglob("*")
              if p.suffix.lower() in {".png", ".jpg"} and "side" in p.parent.name.lower()]
    origen.sort(key=lambda p: int(p.stem) if p.stem.isdigit() else 0)

    for p in origen:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  no se pudo leer: {p.name}")
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (ANCHO, int(h * ANCHO / w)), interpolation=cv2.INTER_AREA)
        guardar(SALIDA / f"{int(p.stem):03d}.jpg", img)

    n = len(list(SALIDA.glob("*.jpg")))
    print(f"{n} imagenes en {SALIDA}")
    print("Subilas a CVAT o Roboflow como un proyecto de keypoints de 7 puntos.")


if __name__ == "__main__":
    main()
