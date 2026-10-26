"""Dataset propio, etapa 5: ¿qué tan bien estima el estimador ACTUAL (entrenado en Bangladesh) sobre animales argentinos?

Usa las predicciones ya calculadas en 42a (pred_peso_v1, con espejado TTA) sobre las fotos `incluida` del dataset curado:
no vuelve a correr ningún modelo. Una predicción por animal = promedio geométrico de sus fotos (como el ensemble de
fotos de RECOLECCION_DATASET.md §2).

Reporta SIEMPRE junto a la referencia obvia: predecir la mediana del propio dataset (ignora la foto).

  A) cero adaptación: el modelo tal cual
  B) recalibración lineal sobre el log-peso (2 parámetros) con validación cruzada POR ANIMAL

Uso:  uv run python 44_propio_evaluar_v1.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

import config


def mape(real, pred):
    return float(np.mean(np.abs(real - pred) / real) * 100)


def main():
    fv = pd.read_csv(config.PROPIO / "fotos.csv", dtype={"animal_id": str})
    ft = pd.read_csv(config.DATA / "propio" / "features_calidad.csv")[["foto", "pred_peso_v1"]]
    d = fv[fv.uso == "incluida"].merge(ft, on="foto")
    a = d.groupby("animal_id").agg(real=("peso_kg", "first"), split=("split", "first"), dia=("dia", "first"),
                                   logp=("pred_peso_v1", lambda s: np.log(s).mean()), n=("foto", "size")).reset_index()
    a["pred"] = np.exp(a.logp)
    print(f"{len(d)} fotos, {len(a)} animales | real {a.real.min():.0f}-{a.real.max():.0f} kg (mediana {a.real.median():.0f})")
    print(f"predicción del modelo actual: {a.pred.min():.0f}-{a.pred.max():.0f} kg (mediana {a.pred.median():.0f})\n")

    # --- A) cero adaptación ---
    med_tr = a[a.split == "train"].real.median()
    filas = []
    for nombre, g in [("todos", a), ("train", a[a.split == "train"]), ("val", a[a.split == "val"]), ("test", a[a.split == "test"]),
                      ("8/10", a[a.dia == "8-10"]), ("22/10", a[a.dia == "22-10"])]:
        filas.append({"grupo": nombre, "animales": len(g), "MAPE_modelo": mape(g.real, g.pred),
                      "MAPE_mediana(train)": mape(g.real, med_tr),
                      "sesgo_%": float(((g.pred - g.real) / g.real).mean() * 100),
                      "spearman": spearmanr(g.real, g.pred)[0]})
    print("A) Modelo actual SIN adaptar vs. predecir la mediana:")
    print(pd.DataFrame(filas).round(1).to_string(index=False))

    # --- B) recalibración lineal en log, CV por animal ---
    x, y = a[["logp"]].to_numpy(), np.log(a.real.to_numpy())
    oof = np.zeros(len(a))
    for tr, te in GroupKFold(n_splits=5).split(x, y, groups=a.animal_id):
        oof[te] = LinearRegression().fit(x[tr], y[tr]).predict(x[te])
    oof_med = np.zeros(len(a))
    for tr, te in GroupKFold(n_splits=5).split(x, y, groups=a.animal_id):
        oof_med[te] = np.median(a.real.to_numpy()[tr])
    print(f"\nB) Recalibración lineal (log-peso) con CV 5 folds por animal, n={len(a)}:")
    print(f"   modelo recalibrado : MAPE {mape(a.real.to_numpy(), np.exp(oof)):.1f}%")
    print(f"   mediana (referencia): MAPE {mape(a.real.to_numpy(), oof_med):.1f}%")
    print(f"   correlación (Spearman) predicción cruda vs. real: {spearmanr(a.real, a.pred)[0]:+.2f}")

    a.to_csv(config.RUNS / "propio_eval_v1_por_animal.csv", index=False)


if __name__ == "__main__":
    main()
