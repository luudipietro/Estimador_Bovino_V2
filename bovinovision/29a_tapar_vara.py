"""Tapa la VARA GRADUADA amarilla de las fotos de CID, por el mismo motivo por el que se tapó el sticker.

Las fotos de CID se toman en una estación fija con una vara de medición amarilla junto al animal. Es un objeto
de tamaño conocido y constante: si entrenamos con esas imágenes tal cual, la CNN puede usarla como referencia de
escala y el resultado dejaría de valer para el escenario real (una foto en el corral, sin nada de referencia).

Detección por color (HSV: amarillo intenso) + componentes conexas grandes. Se excluyen las componentes que caen
dentro del cuerpo del animal, para no borrar el cabestro amarillo que llevan algunos. Después, inpainting con
dilatación aleatoria, igual que en 20a_tapar_sticker.py, para que el tamaño de la zona tapada tampoco informe.

Entrada:  data/pose/extra/<split>/          (CID y Mendeley, generados por 29_dataset_unificado.py)
Salida:   data/pose/extra_sin_vara/<split>/ (CID con la vara tapada; Mendeley se copia igual, no tiene vara)
Después hay que apuntar el manifest a la carpeta nueva (lo hace este mismo script).
Uso:  uv run python 29a_tapar_vara.py
"""
import shutil

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import config

# El cabestro amarillo que llevan algunos animales ronda unos cientos de píxeles; la vara, varios miles.
# Se separa por tamaño y no por posición: la vara suele pasar POR DETRÁS del animal, así que filtrarla por el
# rectángulo del cuerpo (como se intentó primero) dejaba sin tapar justamente el tramo de abajo, el graduado.
AREA_MIN = 1200
DILATAR_MIN, DILATAR_MAX = 9, 25


def mascara_vara(im_bgr):
    """Píxeles de la vara graduada (amarillo intenso, componentes grandes)."""
    hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((H >= 20) & (H <= 35) & (S > 150) & (V > 120)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    sel = np.zeros_like(m)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] >= AREA_MIN:
            sel[lab == k] = 1
    return sel


def main():
    man = pd.read_parquet(config.TABLAS / "manifest_unificado.parquet")
    externos = man[man.dir_imagenes == "extra"]
    if externos.empty:
        print("nada que hacer: el manifest ya apunta a extra_sin_vara")
        return

    tapadas, copiadas, sin_deteccion = 0, 0, 0
    for r in externos.itertuples():
        nombre = r.image_id.replace("/", "__")
        origen = config.DATA / "pose" / "extra" / r.split / nombre
        destino = config.DATA / "pose" / "extra_sin_vara" / r.split / nombre
        destino.parent.mkdir(parents=True, exist_ok=True)
        if destino.exists():
            continue
        if r.origen != "cid":
            shutil.copy2(origen, destino)
            copiadas += 1
            continue

        im = cv2.imread(str(origen))
        m = mascara_vara(im)
        if not m.any():
            shutil.copy2(origen, destino)
            sin_deteccion += 1
            continue
        rng = np.random.default_rng(abs(hash(r.image_id)) % (2 ** 32))
        d = int(rng.integers(DILATAR_MIN, DILATAR_MAX))
        m = cv2.dilate(m, np.ones((d, d), np.uint8))
        limpia = cv2.inpaint(im, m, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        Image.fromarray(cv2.cvtColor(limpia, cv2.COLOR_BGR2RGB)).save(destino, quality=93)
        tapadas += 1

    man.loc[man.dir_imagenes == "extra", "dir_imagenes"] = "extra_sin_vara"
    man.to_parquet(config.TABLAS / "manifest_unificado.parquet", index=False)
    print(f"vara tapada en {tapadas} imágenes de CID | sin detección (se copian igual): {sin_deteccion} | "
          f"Mendeley copiado: {copiadas}")
    print("manifest_unificado.parquet ahora apunta a extra_sin_vara")


if __name__ == "__main__":
    main()
