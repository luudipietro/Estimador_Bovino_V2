"""Etapa 7: Modelo B y referencias, sobre keypoints anotados a mano (techo de la etapa de regresión).

Conjuntos de features:
  d_px     : distancias en píxeles (sin escala, ablación)
  r_ratios : razones entre distancias (sin marcador)                -> Modelo B
  s_sticker: distancias en diámetros de sticker (con marcador físico) -> cota superior con calibración
Modelos: mediana | Ridge | HistGradientBoosting, sobre log(peso). Se entrena en train y se mide en val y test.
Diagnóstico de fuga: mismo modelo con KFold por imagen vs GroupKFold por animal.
Uso:  uv run python 09_baselines.py [--features features_pred]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from common import metrics

FAMILIAS = {"d_px": "d_", "r_ratios": "r_", "s_sticker": "s_"}


def modelo(nombre):
    if nombre == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         l2_regularization=1.0, random_state=config.SEED)


def ajustar_predecir(nombre, Xtr, ytr, Xte):
    m = modelo(nombre).fit(Xtr, np.log(ytr))
    return np.exp(m.predict(Xte))


def cols(df, prefijo):
    return [c for c in df.columns if c.startswith(prefijo)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_gt", help="tabla en data/tablas (sin extensión)")
    a = ap.parse_args()

    df = pd.read_parquet(config.TABLAS / f"{a.features}.parquet")
    df = df.dropna(subset=cols(df, "s_") + cols(df, "r_")).reset_index(drop=True)
    print(f"[{a.features}] {len(df)} imágenes con todas las features")
    tr = df[df.split == "train"]
    filas = []
    for split in ("val", "test"):
        te = df[df.split == split]
        filas.append({"features": "-", "modelo": "mediana", "split": split,
                      **metrics.reporte(tr.peso_kg, te.peso_kg, metrics.baseline_mediana(tr.peso_kg, te.peso_kg), "mediana")})
        for fam, pref in FAMILIAS.items():
            c = cols(df, pref)
            for nom in ("ridge", "hgb"):
                p = ajustar_predecir(nom, tr[c], tr.peso_kg, te[c])
                filas.append({"features": fam, "modelo": nom, "split": split,
                              **metrics.reporte(tr.peso_kg, te.peso_kg, p, nom)})
    res = pd.DataFrame(filas).drop(columns=["modelo"] * 0)
    res = res[["split", "features", "modelo", "n", "mape", "mape_mediana", "mejora_pct"]]
    res.to_csv(config.RUNS / f"baselines_{a.features.replace('features_', '')}.csv", index=False)
    print(res.round(2).to_string(index=False))

    # MAPE por banda del mejor modelo con y sin sticker (test)
    te = df[df.split == "test"]
    for fam in ("r_ratios", "s_sticker"):
        c = cols(df, FAMILIAS[fam])
        p = ajustar_predecir("hgb", tr[c], tr.peso_kg, te[c])
        print(f"\nMAPE por banda (test, hgb, {fam}):")
        print(metrics.mape_por_banda(te.peso_kg, p).round(2).to_string(index=False))

    # Diagnóstico: ¿importa agrupar por animal? (5 folds, hgb sobre ratios, todo el dataset)
    c = cols(df, "r_")
    for nombre, cv in (("KFold por imagen (con fuga potencial)", KFold(5, shuffle=True, random_state=config.SEED)),
                       ("GroupKFold por animal_id", GroupKFold(5))):
        pred = np.zeros(len(df))
        for a, b in cv.split(df, groups=df.animal_id):
            pred[b] = ajustar_predecir("hgb", df.loc[a, c], df.loc[a, "peso_kg"], df.loc[b, c])
        print(f"CV5 {nombre}: MAPE {metrics.mape(df.peso_kg, pred):.2f}%")


if __name__ == "__main__":
    main()
