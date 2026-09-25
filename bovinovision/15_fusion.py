"""Etapa 12: fusión de los modelos A (visual puro) y C (geometría + sticker + forma) -> peso (sec. 12 del plan).

Stacking simple: un segundo regresor (Ridge) toma como entrada [log(pred_A), log(pred_C)] entrenado en TRAIN con
predicciones out-of-fold (5-fold), para no filtrar información del propio ajuste de A/C. Así, si A no aporta nada
nuevo, el stacking puede aprender a ignorarlo (peso ~0); si aporta información complementaria (un error de
profundidad no debería destruir la estimación si el modelo visual compensa, sec. 12 del docx), el MAPE baja.
Requiere haber corrido antes 13_model_C.py y 14_model_A.py (usa sus .joblib y sus predicciones cacheadas).
Uso:  uv run python 15_fusion.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

import config
from common import metrics

SEMILLA = config.SEED


def oof_predicciones(nombre_modelo, X, y, df_split):
    """Predicciones out-of-fold sobre TRAIN (evita que el stacking vea predicciones "de memoria")."""
    import importlib
    bl = importlib.import_module("09_baselines")
    tr_mask = (df_split == "train").to_numpy()
    pred = np.full(len(X), np.nan)
    idx_tr = np.flatnonzero(tr_mask)
    for a, b in KFold(5, shuffle=True, random_state=SEMILLA).split(idx_tr):
        ia, ib = idx_tr[a], idx_tr[b]
        pred[ib] = bl.ajustar_predecir(nombre_modelo, X.iloc[ia], y.iloc[ia], X.iloc[ib])
    return pred


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--escala", choices=["animal", "sticker"], default="animal")
    args = ap.parse_args()

    a = pd.read_parquet(config.TABLAS / "pred_modelo_a.parquet").set_index("image_id")
    c = pd.read_parquet(config.TABLAS / "pred_modelo_c.parquet").set_index("image_id")
    df = a[["split", "peso_kg", "pred_modelo_a"]].join(c[["pred_modelo_c"]], how="inner")
    print(f"{len(df)} imágenes con predicciones de A y C")

    # Predicciones out-of-fold en train para A y C, para entrenar el stacker sin fuga
    import importlib
    bl = importlib.import_module("09_baselines")
    emb_a = pd.read_parquet(config.CACHE / "embeddings_resnet18.parquet").set_index("image_id").loc[df.index]
    ce = [x for x in emb_a.columns if x.startswith("e")]
    from importlib import import_module
    mc = import_module("13_model_C")
    feat_c_full = mc.armar_features("features_pred", args.escala).loc[df.index]
    c_cols = mc.columnas_modelo_c(feat_c_full, args.escala)
    tipo_c = joblib.load(config.RUNS / "models" / "model_c.joblib").get("tipo", "hgb")

    oof_a = oof_predicciones("ridge", emb_a[ce], df.peso_kg, df.split)
    oof_c = oof_predicciones(tipo_c, feat_c_full[c_cols], df.peso_kg, df.split)

    X_stack = pd.DataFrame({"log_a": np.log(np.where(df.split == "train", oof_a, df.pred_modelo_a)),
                            "log_c": np.log(np.where(df.split == "train", oof_c, df.pred_modelo_c))}, index=df.index)
    tr = df.split == "train"
    stacker = RidgeCV(alphas=np.logspace(-3, 2, 20)).fit(X_stack[tr], np.log(df.peso_kg[tr]))
    print(f"pesos del stacker: log_a={stacker.coef_[0]:.3f} log_c={stacker.coef_[1]:.3f} (intercepto {stacker.intercept_:.3f})")

    df["pred_fusion"] = np.exp(stacker.predict(X_stack))
    filas = []
    for split in ("val", "test"):
        m = df.split == split
        for nombre, col in (("A_visual", "pred_modelo_a"), ("C_geom+sticker", "pred_modelo_c"), ("fusion_A+C", "pred_fusion")):
            filas.append({"split": split, "modelo": nombre,
                          **metrics.reporte(df.peso_kg[tr], df.peso_kg[m], df[col][m], nombre)})
    res = pd.DataFrame(filas)[["split", "modelo", "n", "mape", "mape_mediana", "mejora_pct"]]
    res.to_csv(config.RUNS / "fusion.csv", index=False)
    print(res.round(2).to_string(index=False))

    te = df[df.split == "test"]
    print(f"\nMAPE por banda (test, fusión A+C):")
    print(metrics.mape_por_banda(te.peso_kg, te.pred_fusion).round(2).to_string(index=False))

    df.reset_index()[["image_id", "split", "peso_kg", "pred_modelo_a", "pred_modelo_c", "pred_fusion"]].to_parquet(
        config.TABLAS / "predicciones_finales.parquet", index=False)
    joblib.dump(stacker, config.RUNS / "models" / "fusion_stacker.joblib")


if __name__ == "__main__":
    main()
