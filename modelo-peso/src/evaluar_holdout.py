"""Mide el MAPE real de punta a punta, sin la trampa de mezclar animales que
el modelo de pose ya vio en el entrenamiento.

medidas_desde_keypoints.py evalua sobre las 72 fotos juntas, pero 58 de esas
72 las uso YOLO-pose para entrenar -- en esas el modelo memoriza, no predice
de verdad. Este script separa: entrena el regresor solo con los animales de
data/pose/labels/train/, y mide el MAPE solo sobre los de data/pose/labels/val/
(animales que ni el modelo de pose ni el regresor vieron nunca).

Uso (siempre parado en modelo-peso/):
  python src/evaluar_holdout.py --pesos runs/pose7/weights/best.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medidas_desde_keypoints import desde_modelo, distancias, a_ratios
from unir import leer_medidas

import pandas as pd


def main(args):
    val_ids = {int(p.stem) for p in Path("data/pose/labels/val").glob("*.txt")}
    train_ids = {int(p.stem) for p in Path("data/pose/labels/train").glob("*.txt")}
    print(f"train (usados para el modelo de pose): {len(train_ids)} animales")
    print(f"val   (nunca vistos por el modelo):     {len(val_ids)} animales\n")

    filas = []
    for aid, kp in desde_modelo(args.pesos):
        f = {"animal_id": aid}
        d = distancias(np.asarray(kp, float))
        f.update(d)
        f.update(a_ratios(d))
        filas.append(f)

    df = pd.DataFrame(filas).merge(leer_medidas()[["animal_id", "peso_kg"]], on="animal_id")
    cols_r = [c for c in df.columns if c.startswith("r_")]

    tr = df[df["animal_id"].isin(train_ids)]
    va = df[df["animal_id"].isin(val_ids)]
    Xtr, ytr = tr[cols_r].to_numpy(float), tr["peso_kg"].to_numpy(float)
    Xva, yva = va[cols_r].to_numpy(float), va["peso_kg"].to_numpy(float)

    base = np.mean(np.abs((yva - np.median(ytr)) / yva)) * 100
    print(f"Entrenando regresor con {len(tr)}, evaluando en {len(va)} nunca vistos")
    print(f"Baseline (mediana del train): {base:.2f}%\n")

    for nombre, m in [
        ("Ridge", make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 40)))),
        ("Boosting", HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                    max_iter=400, min_samples_leaf=5,
                                                    l2_regularization=1.0, random_state=0)),
    ]:
        m.fit(Xtr, ytr)
        p = m.predict(Xva)
        mape = np.mean(np.abs((yva - p) / yva)) * 100
        mae = np.abs(yva - p).mean()
        print(f"  {nombre:9s} MAPE {mape:5.2f}%   MAE {mae:5.1f} kg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default="runs/pose7/weights/best.pt")
    main(ap.parse_args())
