"""Entrena y valida la estimacion de peso. Particion por ANIMAL, nunca por imagen.

Compara tres cosas para saber donde esta el error:
  1. Schaeffer  - la formula clasica sobre las medidas reales (piso honesto)
  2. Morfometrico - regresion sobre las medidas reales en cm (techo alcanzable)
  3. Escala-libre - regresion sobre la forma de la silueta (lo que nos interesa)

Uso:  python src/entrenar.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

FRANJAS = [(0, 200), (200, 300), (300, 400), (400, 550), (550, 10_000)]


def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


def reportar(nombre, y, pred):
    print(f"\n  {nombre}")
    print(f"    MAPE {mape(y, pred):5.2f}%   MAE {np.abs(y-pred).mean():6.1f} kg   "
          f"RMSE {np.sqrt(((y-pred)**2).mean()):6.1f} kg   "
          f"R2 {1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum():5.3f}")
    for lo, hi in FRANJAS:
        s = (y >= lo) & (y < hi)
        if s.sum() >= 4:
            print(f"      {lo:4d}-{hi if hi < 9999 else '+':>4} kg  n={s.sum():3d}  "
                  f"MAPE {mape(y[s], pred[s]):5.2f}%")
    return mape(y, pred)


def evaluar(X, y, grupos, etiqueta):
    if X.shape[1] == 0 or len(X) == 0:
        print(f"\n  {etiqueta}: sin columnas disponibles, salteado")
        return {}
    cv = GroupKFold(n_splits=min(5, len(np.unique(grupos))))
    modelos = {
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 40))),
        "Boosting": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=400, min_samples_leaf=5,
            l2_regularization=1.0, random_state=0),
    }
    print(f"\n=== {etiqueta} ({X.shape[1]} features, {len(X)} muestras, "
          f"{len(np.unique(grupos))} animales) ===")
    salida = {}
    for nombre, m in modelos.items():
        pred = cross_val_predict(m, X, y, cv=cv, groups=grupos)
        salida[nombre] = (reportar(nombre, y, pred), pred)
    return salida


def main():
    ruta = config.TABLAS / "dataset.csv"
    if not ruta.exists():
        print(f"Falta {ruta}. Corre primero src/unir.py")
        return

    df = pd.read_csv(ruta)
    y = df["peso_kg"].to_numpy(float)
    grupos = df["animal_id"].to_numpy()

    print(f"{len(df)} filas | {df['animal_id'].nunique()} animales")
    print(f"Peso: min {y.min():.0f}  max {y.max():.0f}  media {y.mean():.0f}  sd {y.std():.0f} kg")

    cols_medidas = [c for c in ("perimetro_toracico_cm", "largo_corporal_cm",
                                "altura_cruz_cm", "largo_grupa_cm") if c in df]
    cols_forma = [c for c in df.columns if c.startswith(
        ("sup_", "inf_", "esp_", "hu_", "relacion_", "extent", "solidez", "elongacion",
         "circularidad", "perim_norm", "centroide_", "area_", "espesor_", "topline_"))]

    resultados = {}

    # 0. el baseline que hay que superar para que el MAPE signifique algo:
    #    un modelo que no mira nada y predice siempre la mediana.
    print("\n=== 0. Baseline tonto (predice la mediana, ignora la imagen) ===")
    dummy = np.full_like(y, np.median(y))
    resultados["dummy"] = reportar("Mediana constante", y, dummy)

    # 1. piso: formula de Schaeffer sobre las medidas reales
    if {"perimetro_toracico_cm", "largo_corporal_cm"} <= set(df.columns):
        pred = df["perimetro_toracico_cm"] ** 2 * df["largo_corporal_cm"] / 10838
        print("\n=== 1. Schaeffer (formula clasica, sin entrenar) ===")
        resultados["schaeffer"] = reportar("Schaeffer", y, pred.to_numpy())

    # 2. techo: regresion sobre las medidas reales en cm
    if cols_medidas:
        resultados["morfometrico"] = evaluar(
            df[cols_medidas].to_numpy(float), y, grupos,
            "2. Morfometrico (medidas reales en cm - TECHO)")

    # 3. lo que nos interesa: solo forma, sin escala
    if cols_forma:
        resultados["forma"] = evaluar(
            df[cols_forma].to_numpy(float), y, grupos,
            "3. Escala-libre (forma de la silueta - SIN centimetros)")

    # 4. forma + tamano aparente en px (valido solo si la distancia de captura
    #    esta acotada por protocolo, como en este dataset: 1 m fijo)
    cols_px = [c for c in df.columns if c.startswith("px_")]
    if cols_px:
        resultados["px"] = evaluar(
            df[cols_px].to_numpy(float), y, grupos,
            "4. Solo tamano aparente en px (distancia de captura fija)")
        resultados["forma_px"] = evaluar(
            df[cols_forma + cols_px].to_numpy(float), y, grupos,
            "5. Forma + tamano aparente (lo que daria un protocolo de distancia)")

    print("\n" + "=" * 74)
    print(f"Baseline tonto (mediana): {resultados['dummy']:.2f}%")
    print("Objetivo del acta: MAPE < 15%  |  Techo del 2D segun industria: ~8,6%")
    if resultados["dummy"] < 15:
        print("\nOJO: el baseline tonto YA cumple el objetivo del acta. En este rango de")
        print("pesos, el MAPE solo no prueba nada: hay que reportar la mejora sobre el")
        print("baseline, y el dataset propio tiene que cubrir un rango mas amplio.")
    print("=" * 74)


if __name__ == "__main__":
    main()
