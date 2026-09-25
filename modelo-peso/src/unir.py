"""Cruza los descriptores de forma con las medidas y el peso de bascula.

Uso:  python src/unir.py [--vista side|back]
"""
import argparse
import sys
from pathlib import Path

import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

COLUMNAS = {
    "Num": "animal_id",
    "Oblique body length (cm)": "largo_corporal_cm",
    "Withers height(cm)": "altura_cruz_cm",
    "Heart girth(cm)": "perimetro_toracico_cm",
    "Hip length (cm)": "largo_grupa_cm",
    "Body weight (kg)": "peso_kg",
}


def leer_medidas():
    ruta = next(config.CRUDO.rglob("measurements.xlsx"))
    ws = openpyxl.load_workbook(ruta, data_only=True).active
    filas = list(ws.iter_rows(values_only=True))
    cabecera = [c for c in filas[0] if c is not None]
    datos = [f[:len(cabecera)] for f in filas[1:] if isinstance(f[0], int)]
    df = pd.DataFrame(datos, columns=cabecera).rename(columns=COLUMNAS)
    df["animal_id"] = df["animal_id"].astype(int)
    return df


def main(vista):
    feats = pd.read_csv(config.TABLAS / "features.csv")
    feats[["vista", "animal_id"]] = feats["imagen"].str.split("_", n=1, expand=True)
    feats["animal_id"] = feats["animal_id"].astype(int)
    feats = feats[feats["vista"] == vista].drop(columns=["vista", "imagen"])

    medidas = leer_medidas()
    df = feats.merge(medidas, on="animal_id", how="inner")

    salida = config.TABLAS / "dataset.csv"
    df.to_csv(salida, index=False)
    print(f"vista '{vista}': {df['animal_id'].nunique()} animales, {len(df)} filas "
          f"({len(df)//max(df['animal_id'].nunique(),1)} orientaciones c/u)")
    print(f"peso {df['peso_kg'].min():.0f}-{df['peso_kg'].max():.0f} kg")
    print(f"-> {salida}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vista", default="side", choices=["side", "back"])
    main(ap.parse_args().vista)
