"""Dataset de prueba, etapa 1: hojas de revisión para leer peso e IDE de las pantallas del Tru-Test (6/10).

Mismo método que `40_pesajes_leer.py` (las pantallas se leen a ojo, no hay OCR) sobre `dataset_prueba/martes_6_10_26/`.
Salida: data/prueba/revision/pantallas_NN.jpg y data/prueba/pantallas.csv (archivo, hora, orden).
Uso:  uv run --extra cpu python 50_prueba_leer.py
"""

import importlib

import pandas as pd
from PIL import Image, ImageDraw

import config

base = importlib.import_module("40_pesajes_leer")  # enderezar() y hora_exif()
CARPETA = "martes_6_10_26"
SALIDA = config.DATA / "prueba"


def main():
    hojas = SALIDA / "revision"
    hojas.mkdir(parents=True, exist_ok=True)
    fotos = sorted((config.PRUEBA / CARPETA).glob("*.jpg"), key=lambda p: (base.hora_exif(p), p.name))
    lado = base.LADO
    for h in range(0, len(fotos), base.POR_HOJA):
        hoja = Image.new("RGB", (2 * lado, 2 * (lado * 3 // 4 + 22)), "white")
        d = ImageDraw.Draw(hoja)
        for k, p in enumerate(fotos[h:h + base.POR_HOJA]):
            with Image.open(p) as im:
                im = base.enderezar(im)
                im.thumbnail((lado, lado * 3 // 4))
                x, y = (k % 2) * lado, (k // 2) * (lado * 3 // 4 + 22)
                d.text((x + 4, y + 4), f"#{h + k + 1}  {p.name}", fill="black")
                hoja.paste(im, (x, y + 22))
        hoja.save(hojas / f"pantallas_{h // base.POR_HOJA + 1:02d}.jpg", quality=88)
    pd.DataFrame([{"archivo": p.name, "hora": base.hora_exif(p), "orden": i} for i, p in enumerate(fotos, 1)]).to_csv(
        SALIDA / "pantallas.csv", index=False)
    print(f"{len(fotos)} pantallas -> {hojas}")


if __name__ == "__main__":
    main()
