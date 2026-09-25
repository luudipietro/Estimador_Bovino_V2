"""Fase 2: combinar la CNN (Fase 1) con el camino geométrico (Modelo C', ratios de keypoints + forma).

La CNN ya es el mejor modelo por lejos. Acá se prueba si la geometría todavía aporta algo que la imagen no
capture, como planteaba la sec. 12 del plan (dos caminos, visual y geométrico).

Detalle metodológico importante: las predicciones de la CNN sobre el split de train NO sirven para ajustar el
combinador (la CNN entrenó con esas imágenes y las predice casi perfecto). Por eso el combinador se ajusta sobre
VALIDACIÓN —que la CNN nunca vio— y se mide en TEST. La contra es que val también se usó para elegir la mejor
corrida, así que hay algo de sobreajuste a val; por eso se reporta siempre el número de test.

Uso:  uv run python 25_fusion_cnn.py [--top 3]
"""
import argparse
import importlib

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

import config
from common import metrics

bl = importlib.import_module("09_baselines")
mc = importlib.import_module("13_model_C")


def ensemble_cnn(top: int) -> pd.DataFrame:
    """Promedio geométrico de las mejores corridas SIN sticker, elegidas por su MAPE de validación."""
    r = pd.read_csv(config.RUNS / "cnn" / "resumen.csv")
    r = r[r.imagenes == "sin_sticker"].sort_values("mape_val").head(top)
    dfs = [pd.read_parquet(config.TABLAS / f"pred_cnn_{n}.parquet").set_index("image_id") for n in r.nombre]
    base = dfs[0][["split", "peso_kg"]].copy()
    base["pred_cnn"] = np.exp(np.mean([np.log(d.pred_cnn.loc[base.index]) for d in dfs], axis=0))
    print(f"ensemble de {len(dfs)}: {', '.join(r.nombre)}")
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()

    cnn = ensemble_cnn(a.top)

    # camino geométrico (sin marcador): se entrena en TRAIN como siempre
    feat = mc.armar_features("features_pred", "animal")
    cols_c = mc.columnas_modelo_c(feat, "animal")
    feat = feat.dropna(subset=cols_c)
    tr = feat[feat.split == "train"]
    y_tr = tr.peso_kg.copy()  # se guarda antes de filtrar: cnn solo tiene val/test, no train
    idx = cnn.index.intersection(feat.index)
    cnn, feat = cnn.loc[idx], feat.loc[idx]
    pred_c = bl.ajustar_predecir("ridge", tr[cols_c], tr.peso_kg, feat[cols_c])

    df = cnn.copy()
    df["pred_c"] = pred_c
    val, test = df[df.split == "val"], df[df.split == "test"]

    print(f"\n{len(df)} imágenes | val {len(val)} | test {len(test)}")
    print("\n=== componentes (test) ===")
    for nombre, col in (("CNN (ensemble)", "pred_cnn"), ("C' geometría", "pred_c")):
        print(f"  {nombre:16s} {metrics.mape(test.peso_kg, test[col]):.2f}%")

    # combinador ajustado en validación
    X = lambda d: np.column_stack([np.log(d.pred_cnn), np.log(d.pred_c)])
    comb = RidgeCV(alphas=np.logspace(-3, 3, 30)).fit(X(val), np.log(val.peso_kg))
    print(f"\npesos del combinador: cnn={comb.coef_[0]:.3f} geometría={comb.coef_[1]:.3f}")
    df["pred_fusion"] = np.exp(comb.predict(X(df)))
    test = df[df.split == "test"]

    filas = []
    for nombre, col in (("CNN sola", "pred_cnn"), ("C' geometría sola", "pred_c"), ("fusión CNN+C'", "pred_fusion")):
        filas.append({"modelo": nombre, **metrics.reporte(y_tr, test.peso_kg, test[col], nombre)})
    res = pd.DataFrame(filas)[["modelo", "n", "mape", "mape_mediana", "mejora_pct"]]
    print("\n=== test ===")
    print(res.round(2).to_string(index=False))
    res.to_csv(config.RUNS / "fusion_cnn.csv", index=False)

    mejor_col = "pred_fusion" if res.set_index("modelo").loc["fusión CNN+C'", "mape"] < res.set_index("modelo").loc["CNN sola", "mape"] else "pred_cnn"
    print(f"\nMAPE por banda (test, {mejor_col}):")
    print(metrics.mape_por_banda(test.peso_kg, test[mejor_col]).round(2).to_string(index=False))
    df.reset_index().to_parquet(config.TABLAS / "predicciones_sin_sticker.parquet", index=False)
    print(f"-> {config.TABLAS / 'predicciones_sin_sticker.parquet'}")


if __name__ == "__main__":
    main()
