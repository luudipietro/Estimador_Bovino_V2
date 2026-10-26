"""Dataset propio, etapa 6b: tabla comparativa de todas las corridas de 46_propio_cv.py (runs/propio_cv/*/resumen.json).

Cada corrida es el resultado OUT-OF-FOLD sobre los 141 animales: ningún animal se evalúa con un modelo que lo haya visto.
Si hay varias semillas de una misma configuración las agrupa (media y desvío) y además calcula el ENSEMBLE de semillas
(promedio geométrico de las predicciones por animal), que es lo que se reportaría.

Uso:  uv run python 47_propio_resumen_cv.py
"""
import json
import re

import numpy as np
import pandas as pd

import config


def main():
    filas, por_animal = [], {}
    for d in sorted((config.RUNS / "propio_cv").glob("*/resumen.json")):
        r = json.loads(d.read_text(encoding="utf-8"))
        if r["nombre"] == "humo" or r["animales"] < 100:   # corridas de prueba
            continue
        filas.append({**{k: r[k] for k in ("nombre", "mape_modelo", "mape_mediana", "mejora_puntos", "spearman", "sesgo_pct",
                                           "pred_min", "pred_max")}, "ic95_mejora": f"{r['mejora_ic95'][0]:+.2f} a {r['mejora_ic95'][1]:+.2f}"})
        por_animal[r["nombre"]] = pd.read_csv(d.parent / "oof_por_animal.csv", index_col=0)
    if not filas:
        raise SystemExit("no hay corridas completas en runs/propio_cv/")
    t = pd.DataFrame(filas).sort_values("mape_modelo")
    print("Corridas (OOF, 141 animales; la mediana es la referencia de 'no mirar la foto'):\n")
    print(t.round(2).to_string(index=False))

    # agrupar semillas: nombre_sN -> configuración
    conf = {}
    for n in por_animal:
        conf.setdefault(re.sub(r"_s\d+$", "", n), []).append(n)
    print("\nPor configuración (promedio de semillas y ensemble de semillas):")
    for c, ns in conf.items():
        ind = [t.loc[t.nombre == n, "mape_modelo"].iloc[0] for n in ns]
        base = por_animal[ns[0]]
        ens = np.exp(np.mean([np.log(por_animal[n].pred) for n in ns], axis=0))
        mape_ens = float(np.mean(np.abs(ens - base.real) / base.real) * 100)
        med = float(np.mean(np.abs(base.mediana_train - base.real) / base.real) * 100)
        print(f"  {c:<28} {len(ns)} semilla(s): individual {np.mean(ind):.2f}% ± {np.std(ind):.2f} | ensemble {mape_ens:.2f}% | mediana {med:.2f}%")
    print("\nReferencias: predecir la mediana ~8,4 % | Ridge sobre embedding congelado (45_propio_embeddings.py) ~7,2 % | "
          "modelo actual sin adaptar ~49 %.")


if __name__ == "__main__":
    main()
