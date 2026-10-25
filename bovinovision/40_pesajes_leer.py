"""Dataset propio, etapa 1: hojas de revisión para leer peso e IDE de las fotos de la pantalla del Tru-Test.

El peso real y la caravana (IDE) de cada animal están SOLO dentro de las fotos de la pantalla de la balanza
(`*pesos/`); no hay archivo de texto. Este script ordena esas fotos por la hora EXIF, las endereza (varias se sacaron
con el teléfono vertical) y arma hojas de 4 pantallas con el nombre del archivo, para leerlas y cargarlas en
`dataset_propio/pesajes_leidos.csv`. La lectura se hace a ojo: las pantallas son nítidas y no hay OCR instalado.

Salida: data/propio/revision/<carpeta>_<nn>.jpg   y   data/propio/pantallas.csv (archivo, hora, orden)
Uso:  uv run python 40_pesajes_leer.py
"""
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

import config

CARPETAS = ["8-10pesos", "22-10pesos"]
POR_HOJA = 4          # 2x2
LADO = 720            # px por pantalla en la hoja: alcanza para leer el peso y el IDE
HOJAS = config.DATA / "propio" / "revision"


def hora_exif(ruta):
    with Image.open(ruta) as im:
        return im.getexif().get_ifd(0x8769)[0x9003]  # DateTimeOriginal


def enderezar(im):
    """Aplica la orientación EXIF y, si la pantalla quedó acostada (más alta que ancha), la gira a horizontal."""
    im = ImageOps.exif_transpose(im)
    return im.rotate(-90, expand=True) if im.height > im.width else im


def main():
    HOJAS.mkdir(parents=True, exist_ok=True)
    filas = []
    for carpeta in CARPETAS:
        fotos = sorted((config.PROPIO / carpeta).glob("*.jpg"), key=lambda p: (hora_exif(p), p.name))
        for i, p in enumerate(fotos, 1):
            filas.append({"carpeta": carpeta, "archivo": p.name, "hora": hora_exif(p), "orden": i})
        for h in range(0, len(fotos), POR_HOJA):
            lote = fotos[h:h + POR_HOJA]
            hoja = Image.new("RGB", (2 * LADO, 2 * (LADO * 3 // 4 + 22)), "white")
            d = ImageDraw.Draw(hoja)
            for k, p in enumerate(lote):
                with Image.open(p) as im:
                    im = enderezar(im)
                    im.thumbnail((LADO, LADO * 3 // 4))
                    x, y = (k % 2) * LADO, (k // 2) * (LADO * 3 // 4 + 22)
                    d.text((x + 4, y + 4), f"#{h + k + 1}  {p.name}", fill="black")
                    hoja.paste(im, (x, y + 22))
            hoja.save(HOJAS / f"{carpeta}_{h // POR_HOJA + 1:02d}.jpg", quality=88)
    pd.DataFrame(filas).to_csv(config.DATA / "propio" / "pantallas.csv", index=False)
    print(f"{len(filas)} pantallas -> {HOJAS} ({len(list(HOJAS.glob('*.jpg')))} hojas)")


if __name__ == "__main__":
    main()
