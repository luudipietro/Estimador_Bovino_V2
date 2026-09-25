"""Etapa 8 (previa): ¿cuánto error de escala tolera el modelo de peso? Define el umbral de la puerta de profundidad.

Se parte de las distancias en diámetros de sticker (escala perfecta) y se simula un error de escala por imagen:
   s_ruidoso = s * exp(eps),  eps ~ N(0, sigma)      (mismo factor para las 36 distancias de la imagen)
Dos escenarios:
  A) entrenar con escala perfecta y desplegar con escala ruidosa (lo que pasaría si la profundidad solo se usa en inferencia)
  B) entrenar y evaluar con el mismo nivel de ruido (el modelo aprende a desconfiar de la escala)
Features: 's' (solo escala) y 's+r' (escala + ratios sin escala, el modelo puede ignorar la escala si es mala).
Referencias: mediana, Modelo B (ratios, sin marcador) y escala perfecta.
Limitación: simula un error de escala GLOBAL por imagen; la profundidad real también deforma las proporciones
entre puntos (error no uniforme), que esto no captura -> el umbral resultante es optimista.
Uso:  uv run python 10a_sensibilidad_escala.py
"""
import importlib

import numpy as np
import pandas as pd

import config
from common import metrics

bl = importlib.import_module("09_baselines")

SIGMAS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
SEMILLAS = [0, 1, 2, 3, 4]


def ruido(df, cols_s, sigma, seed):
    """Copia del df con las columnas s_* multiplicadas por exp(eps) por imagen."""
    out = df.copy()
    eps = np.random.default_rng(seed).normal(0, sigma, len(df))
    out[cols_s] = out[cols_s].to_numpy() * np.exp(eps)[:, None]
    return out


def main():
    df = pd.read_parquet(config.TABLAS / "features_gt.parquet")
    df = df.dropna(subset=bl.cols(df, "s_") + bl.cols(df, "r_")).reset_index(drop=True)
    cs, cr = bl.cols(df, "s_"), bl.cols(df, "r_")
    tr, te = df[df.split == "train"], df[df.split == "test"]
    y_tr, y_te = tr.peso_kg, te.peso_kg

    med = metrics.mape(y_te, metrics.baseline_mediana(y_tr, y_te))
    modelo_b = metrics.mape(y_te, bl.ajustar_predecir("hgb", tr[cr], y_tr, te[cr]))
    print(f"Referencias en test: mediana {med:.2f}% | Modelo B sin marcador {modelo_b:.2f}%")

    filas = []
    for sigma in SIGMAS:
        for esc in ("A_train_limpio", "B_train_ruidoso"):
            for fam, c in (("s", cs), ("s+r", cs + cr)):
                v = []
                for seed in SEMILLAS if sigma > 0 else SEMILLAS[:1]:
                    te_r = ruido(te, cs, sigma, 1000 + seed)
                    tr_r = tr if esc == "A_train_limpio" else ruido(tr, cs, sigma, 2000 + seed)
                    v.append(metrics.mape(y_te, bl.ajustar_predecir("hgb", tr_r[c], y_tr, te_r[c])))
                filas.append({"sigma_escala_pct": int(sigma * 100), "escenario": esc, "features": fam,
                              "mape_media": np.mean(v), "mape_sd": np.std(v)})
                print(f"  sigma={sigma:.0%} {esc:16s} {fam:3s} MAPE {np.mean(v):.2f} ± {np.std(v):.2f}")
    res = pd.DataFrame(filas)
    res.to_csv(config.RUNS / "sensibilidad_escala.csv", index=False)

    print("\nMAPE (test) por error de escala sigma — escenario B (entrena con el mismo ruido):")
    tabla = res[res.escenario == "B_train_ruidoso"].pivot(index="sigma_escala_pct", columns="features", values="mape_media")
    print(tabla.round(2).to_string())
    print("\nMAPE (test) por error de escala sigma — escenario A (entrena limpio, despliega ruidoso):")
    print(res[res.escenario == "A_train_limpio"].pivot(index="sigma_escala_pct", columns="features",
                                                     values="mape_media").round(2).to_string())

    limpio = res[(res.sigma_escala_pct == 0) & (res.features == "s")].mape_media.iloc[0]
    b = res[(res.escenario == "B_train_ruidoso") & (res.features == "s+r")].set_index("sigma_escala_pct").mape_media
    util = [s for s, m in b.items() if m <= limpio + 0.5 * (modelo_b - limpio)]
    print(f"\nEscala perfecta: {limpio:.2f}% | sin marcador (B): {modelo_b:.2f}%")
    print(f"sigma maxima donde la escala recupera al menos la mitad de su beneficio (s+r, escenario B): "
          f"{max(util) if util else 0}%")


if __name__ == "__main__":
    main()
