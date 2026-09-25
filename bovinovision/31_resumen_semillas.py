"""Consolida las corridas con distintas semillas y produce el número final con su incertidumbre.

Combina las dos fuentes de error que medimos por separado:
  - variabilidad del ENTRENAMIENTO -> desvío entre semillas (correr_semillas.sh, en Fedora)
  - variabilidad del TEST          -> bootstrap remuestreando animales (30_intervalos.py, corre en cualquier lado)

También responde si la mejora del balanceo es real: compara las dos familias de semillas con un test de
Mann-Whitney (no paramétrico, adecuado para tan pocas muestras).

Uso:  uv run python 31_resumen_semillas.py
"""
import numpy as np
import pandas as pd

import config
from common import metrics


def main():
    man = pd.read_parquet(config.TABLAS / "manifest.parquet").set_index("image_id")
    ids_te = set(man[man.split == "test"].index)
    r = pd.read_csv(config.RUNS / "cnn" / "resumen.csv")
    if "semilla" not in r.columns or r.semilla.notna().sum() == 0:
        print("Todavía no hay corridas con semilla. Correr antes, en Fedora:\n  bash correr_semillas.sh")
        return

    con_semilla = r[r.semilla.notna()].copy()
    filas = []
    for x in con_semilla.itertuples():
        p = config.TABLAS / f"pred_cnn_{x.nombre}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p).set_index("image_id")
        d = d[(d.split == "test") & d.index.isin(ids_te)]
        filas.append({"familia": "con balanceo" if x.balanceado else "sin balanceo",
                      "semilla": int(x.semilla), "mape_test": metrics.mape(d.peso_kg, d.pred_cnn),
                      "mape_val": x.mape_val})
    if not filas:
        print("No encontré las predicciones de las corridas con semilla.")
        return
    t = pd.DataFrame(filas).sort_values(["familia", "semilla"])
    print(t.round(2).to_string(index=False))

    print("\n=== por configuración ===")
    resumen = t.groupby("familia").mape_test.agg(["count", "mean", "std", "min", "max"]).round(2)
    print(resumen.to_string())

    for fam, g in t.groupby("familia"):
        m, s, n = g.mape_test.mean(), g.mape_test.std(), len(g)
        ic = 1.96 * s / np.sqrt(n)
        print(f"\n{fam}: MAPE {m:.2f}% ± {s:.2f} (desvío entre semillas) | "
              f"IC95 de la media: {m - ic:.2f}-{m + ic:.2f}")

    familias = t.familia.unique()
    if len(familias) == 2:
        from scipy.stats import mannwhitneyu
        a = t[t.familia == familias[0]].mape_test
        b = t[t.familia == familias[1]].mape_test
        p = mannwhitneyu(a, b).pvalue
        print(f"\n¿La diferencia entre '{familias[0]}' y '{familias[1]}' es real? "
              f"Mann-Whitney p={p:.3f} -> {'sí' if p < 0.05 else 'NO se puede afirmar (es ruido)'}")

    boot = config.RUNS / "intervalos_bootstrap.csv"
    if boot.exists():
        b = pd.read_csv(boot)
        print(f"\n=== combinando con la incertidumbre del test (bootstrap por animal) ===")
        print(b.round(2).to_string(index=False))
        print("\nPara el informe, la forma honesta de decirlo:")
        mejor = t.groupby("familia").mape_test.mean().idxmin()
        m = t[t.familia == mejor].mape_test
        fila = b.iloc[0]
        print(f"  MAPE {m.mean():.1f}% (desvío entre semillas ±{m.std():.1f}; "
              f"IC 95% por muestreo de animales: {fila.ic95_desde:.1f}-{fila.ic95_hasta:.1f})")
    t.to_csv(config.RUNS / "resumen_semillas.csv", index=False)
    print(f"\n-> {config.RUNS / 'resumen_semillas.csv'}")


if __name__ == "__main__":
    main()
