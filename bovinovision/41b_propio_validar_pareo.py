"""Dataset propio, etapa 2b: ¿el pareo foto-de-vaca <-> pesaje está bien, o corrido un lugar?

No hay animales repetidos entre días (ningún IDE se repite), así que no se puede validar cruzando pelajes. En su
lugar se usa una señal independiente: el estimador actual (entrenado en Bangladesh, no vio nada de esto) ordena
a los animales por tamaño aunque no acierte la magnitud. Si el pareo es correcto, el peso que predice para las
fotos del pesaje i debe correlacionar mejor con el peso real del pesaje i que con el del vecino i+-1.
La correlación con corrimiento 0 tiene que ser la máxima.

Uso:  uv run python 41b_propio_validar_pareo.py       (necesita data/propio/features_calidad.csv, de 42a)
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config

CORRIMIENTOS = [-3, -2, -1, 0, 1, 2, 3]


def main():
    fv = pd.read_csv(config.PROPIO / "fotos.csv")[["foto", "dia", "pesaje_id", "peso_kg", "peso_confiable"]]
    ft = pd.read_csv(config.DATA / "propio" / "features_calidad.csv")
    d = fv.merge(ft, on="foto")
    d = d[d.pesaje_id.notna() & d.peso_confiable & (d.conf_det >= 0.5) & d.pred_peso_v1.notna()]
    por = d.groupby(["dia", "pesaje_id"]).agg(pred=("pred_peso_v1", "median"), peso=("peso_kg", "first"),
                                              n=("foto", "size")).reset_index()
    print(f"pesajes con predicción: {len(por)} (de {fv.pesaje_id.nunique()})")
    rng = np.random.default_rng(config.SEED)
    for dia, g in list(por.groupby("dia")) + [("ambos", por)]:
        g = g.sort_values("pesaje_id").reset_index(drop=True)
        print(f"\n== {dia} (n={len(g)}): Spearman entre la predicción del pesaje i y el peso real del pesaje i+k")
        for k in CORRIMIENTOS:
            if dia == "ambos":   # corrimientos dentro de cada día, sin cruzar el límite
                pares = [(a, b) for _, h in g.groupby("dia") for a, b in zip(h.pred.iloc[max(0, -k):len(h) - max(0, k)],
                                                                            h.peso.iloc[max(0, k):len(h) - max(0, -k)])]
                x, y = zip(*pares)
            else:
                x = g.pred.iloc[max(0, -k):len(g) - max(0, k)]
                y = g.peso.iloc[max(0, k):len(g) - max(0, -k)]
            r = spearmanr(x, y)[0]
            marca = "  <-- pareo declarado" if k == 0 else ""
            print(f"  k={k:+d}: rho={r:+.3f}{marca}")
        # permutación: ¿cuánto se esperaría por azar?
        azar = [spearmanr(g.pred, rng.permutation(g.peso.to_numpy()))[0] for _ in range(500)]
        print(f"  (azar, 500 permutaciones: media {np.mean(azar):+.3f}, P95 {np.percentile(azar, 95):+.3f})")


if __name__ == "__main__":
    main()
