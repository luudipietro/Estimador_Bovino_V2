"""Fase 1 (paso previo obligatorio): genera las imágenes CON EL STICKER BORRADO.

Por qué: el marcador físico está pegado al animal en casi todas las fotos de AcmeAI. Si entrenamos una CNN sobre
las imágenes tal cual, el modelo puede aprender a usar el sticker como referencia de escala (es un objeto de
tamaño conocido y constante), y entonces el MAPE que midiéramos NO valdría para el escenario real, donde no va a
haber sticker. Para que el experimento "sin marcador" sea honesto, hay que sacarlo de la imagen.

Se usa la máscara del sticker que ya trae el dataset + inpainting de OpenCV (rellena con el entorno: pelo del
animal), sobre las imágenes de 1024 px que usa el entrenamiento.

Salida: data/pose/images_sin_sticker/<split>/<image_id>   (mismo nombre que data/pose/images/)
Uso:  uv run python 20a_tapar_sticker.py
"""
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import config
from common import masks

LADO_POSE = 1024
# Dilatación ALEATORIA por imagen (determinista por image_id): si tapáramos siempre justo el sticker, el tamaño
# de la zona borrosa seguiría siendo proporcional al del marcador y la CNN podría leer la escala de ahí.
DILATAR_MIN, DILATAR_MAX = 9, 31


def procesar(args) -> str:
    """Devuelve 'ok' | 'sin_sticker' | 'sin_imagen' | 'ya_estaba'."""
    image_id, split, ruta_mascara = args
    nombre = image_id.replace("/", "__")
    origen = config.DATA / "pose" / "images" / split / nombre
    destino = config.DATA / "pose" / "images_sin_sticker" / split / nombre
    if destino.exists():
        return "ya_estaba"
    if not origen.exists():
        # 06a_preparar_pose.py solo preparó las imágenes con keypoints anotados (4.540 de 4.544)
        return "sin_imagen"

    im = np.array(Image.open(origen).convert("RGB"))
    rgba = np.array(Image.open(config.ACMEAI / ruta_mascara).convert("RGBA"))
    _, sticker = masks.separar(rgba)
    if not sticker.any():
        Image.fromarray(im).save(destino, quality=93)
        return "sin_sticker"

    m = np.array(Image.fromarray((sticker * 255).astype(np.uint8)).resize((im.shape[1], im.shape[0]), Image.NEAREST))
    rng = np.random.default_rng(abs(hash(image_id)) % (2 ** 32))
    k = int(rng.integers(DILATAR_MIN, DILATAR_MAX))
    m = cv2.dilate(m, np.ones((k, k), np.uint8))
    limpia = cv2.inpaint(im, m, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    Image.fromarray(limpia).save(destino, quality=93)
    return "ok"


def main():
    man = pd.read_parquet(config.TABLAS / "manifest.parquet")
    for split in man.split.unique():
        (config.DATA / "pose" / "images_sin_sticker" / split).mkdir(parents=True, exist_ok=True)

    tareas = [(r.image_id, r.split, r.ruta_mascara) for r in man.itertuples()]
    conteo = {}
    with ProcessPoolExecutor() as pool:
        for i, res in enumerate(pool.map(procesar, tareas, chunksize=16), 1):
            conteo[res] = conteo.get(res, 0) + 1
            if i % 500 == 0:
                print(f"  {i}/{len(tareas)}")
    print(f"listo -> {config.DATA / 'pose' / 'images_sin_sticker'}\n  {conteo}")


if __name__ == "__main__":
    main()
