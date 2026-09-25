"""Etapa 1b: asigna train/val/test POR animal_id y lo escribe en el manifest (columna 'split').
Se ejecuta una sola vez: el split queda fijo (seed en config.py) y lo usan todas las etapas siguientes."""
import pandas as pd

import config
from common import splits


def main():
    ruta = config.TABLAS / "manifest.parquet"
    df = pd.read_parquet(ruta).drop(columns=["split"], errors="ignore")
    mapa = splits.asignar_split(df["animal_id"])
    df["split"] = df["animal_id"].map(mapa)
    splits.verificar_sin_fuga(df)
    df.to_parquet(ruta, index=False)
    print(df.groupby("split").agg(imagenes=("image_id", "size"), animales=("animal_id", "nunique"),
                                  peso_med=("peso_kg", "median")))
    print(df.groupby(["split", "lote"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
