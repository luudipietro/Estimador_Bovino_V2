"""Etapa 1: arma data/tablas/manifest.parquet (una fila por imagen lateral de B3+B4).

Columnas: image_id, lote, subbatch, animal_id, sexo, peso_kg, ruta_imagen, ruta_mascara (relativas al dataset),
ancho, alto, focal_35mm (EXIF, NaN si falta), tiene_mascara.
Uso:  uv run python 01_manifest.py
"""
import pandas as pd
from PIL import Image

import config
from common import acmeai

EXIF_FOCAL_35MM = 0xA405  # FocalLengthIn35mmFilm (sub-IFD Exif)


def leer_meta(ruta):
    """Ancho, alto y focal equivalente 35 mm (o None) leyendo solo la cabecera del JPG."""
    with Image.open(ruta) as im:
        ancho, alto = im.size
        try:
            focal = im.getexif().get_ifd(0x8769).get(EXIF_FOCAL_35MM)
        except Exception:
            focal = None
    return ancho, alto, (float(focal) if focal else None)


def main():
    filas, descartadas = [], []
    for lote, rutas in acmeai.LOTES_SIDE.items():
        d_img = config.ACMEAI / rutas["imagenes"]
        d_msk = config.ACMEAI / rutas["mascaras"]
        for img in sorted(d_img.glob("*.jpg")):
            info = acmeai.parsear_nombre(lote, img.name)
            if info is None:
                descartadas.append((lote, img.name))
                continue
            ancho, alto, focal = leer_meta(img)
            msk = acmeai.ruta_mascara(d_msk, img.name)
            filas.append({
                "image_id": f"{lote}/{img.name}", "lote": lote, **info,
                "ruta_imagen": img.relative_to(config.ACMEAI).as_posix(),
                "ruta_mascara": msk.relative_to(config.ACMEAI).as_posix(),
                "tiene_mascara": msk.exists(),
                "ancho": ancho, "alto": alto, "focal_35mm": focal,
            })
    df = pd.DataFrame(filas)
    assert df["image_id"].is_unique, "image_id repetido"
    salida = config.TABLAS / "manifest.parquet"
    df.to_parquet(salida, index=False)

    print(f"{len(df)} imágenes -> {salida}")
    print(f"descartadas por nombre irregular: {len(descartadas)} {descartadas[:3]}")
    print(df.groupby("lote").agg(n=("image_id", "size"), animales=("animal_id", "nunique"),
                                 peso_min=("peso_kg", "min"), peso_med=("peso_kg", "median"),
                                 peso_max=("peso_kg", "max"), con_mascara=("tiene_mascara", "sum")))
    print(f"resoluciones: {df.groupby(['lote', 'ancho', 'alto']).size().to_dict()}")
    print(f"EXIF focal 35mm presente: {df['focal_35mm'].notna().mean():.1%} "
          f"(valores: {df['focal_35mm'].value_counts().head(3).to_dict()})")


if __name__ == "__main__":
    main()
