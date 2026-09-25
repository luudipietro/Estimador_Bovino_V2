"""Etapa 2: distribución de pesos y baseline de la mediana (el piso contra el que se mide todo).
Uso:  uv run python 03_eda.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from common import metrics


def main():
    df = pd.read_parquet(config.TABLAS / "manifest.parquet")
    tr, te = df[df.split == "train"], df[df.split == "test"]

    print("Pesos (kg) por lote:")
    print(df.groupby("lote").peso_kg.describe().round(1).to_string())
    print("\nPesos (kg) por sexo:")
    print(df.groupby("sexo").peso_kg.describe().round(1).to_string())

    pred = metrics.baseline_mediana(tr.peso_kg, te.peso_kg)
    print(f"\nBaseline mediana (mediana train = {tr.peso_kg.median():.0f} kg) sobre test: "
          f"MAPE {metrics.mape(te.peso_kg, pred):.2f}% | MAE {metrics.mae(te.peso_kg, pred):.1f} kg")
    print("\nMAPE de la mediana por banda de peso (test):")
    tabla = metrics.mape_por_banda(te.peso_kg, pred)
    print(tabla.to_string(index=False))
    tabla.to_csv(config.RUNS / "eda_baseline_mediana_por_banda.csv", index=False)

    print("\nImágenes por banda y split:")
    df["banda"] = pd.cut(df.peso_kg, config.BANDAS_PESO, right=False)
    print(df.groupby(["banda", "split"], observed=True).size().unstack(fill_value=0).to_string())

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for lote, g in df.groupby("lote"):
        ax.hist(g.peso_kg, bins=40, alpha=0.6, label=lote)
    ax.set_xlabel("peso (kg)"); ax.set_ylabel("imágenes"); ax.legend()
    fig.tight_layout(); fig.savefig(config.RUNS / "eda_hist_peso.png", dpi=110)


if __name__ == "__main__":
    main()
