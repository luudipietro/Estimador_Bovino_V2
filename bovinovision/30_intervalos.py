"""Intervalos de confianza del MAPE, para poder reportar "15,1% ± algo" en vez de un número pelado.

Fuente de incertidumbre que cubre este script: **qué animales tocaron en el conjunto de test**. Se estima por
bootstrap remuestreando ANIMALES (no imágenes) con reemplazo: si un animal entra, entran todas sus fotos. Hacerlo
por imagen daría un intervalo artificialmente angosto, porque las fotos del mismo animal están correlacionadas.

No cubre la variabilidad del entrenamiento (inicialización, orden de los lotes); para eso hay que reentrenar con
varias semillas, que es lo que hace `correr_semillas.sh` en Fedora, y después `31_resumen_semillas.py` combina las dos.

Uso:  uv run python 30_intervalos.py [--repeticiones 2000]
"""
import argparse

import numpy as np
import pandas as pd

import config
from common import metrics

MODELOS = {
    "CNN (mejor sola)": "convnext_tiny_crop_384_mse",
    "CNN + balanceo": "convnext_tiny_crop_384_mse_bal",
    "(control) con sticker": "convnext_tiny_crop_384_l1_CONSTICKER",
}
ENSEMBLE = ["convnext_tiny_crop_384_mse", "convnext_tiny_crop_384_mse_bal", "convnext_tiny_crop_384_l1_noaug"]


def bootstrap_por_animal(df, repeticiones, seed=config.SEED):
    """df con columnas animal_id, peso_kg, pred. Devuelve (media, p2.5, p97.5) del MAPE."""
    rng = np.random.default_rng(seed)
    animales = df.animal_id.unique()
    por_animal = {a: g for a, g in df.groupby("animal_id")}
    valores = []
    for _ in range(repeticiones):
        elegidos = rng.choice(animales, size=len(animales), replace=True)
        m = pd.concat([por_animal[a] for a in elegidos])
        valores.append(metrics.mape(m.peso_kg, m.pred))
    v = np.array(valores)
    return v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeticiones", type=int, default=2000)
    a = ap.parse_args()

    man = pd.read_parquet(config.TABLAS / "manifest.parquet").set_index("image_id")
    ids_te = set(man[man.split == "test"].index)

    def cargar(nombre):
        p = pd.read_parquet(config.TABLAS / f"pred_cnn_{nombre}.parquet").set_index("image_id")
        p = p[(p.split == "test") & p.index.isin(ids_te)]
        return p.pred_cnn

    preds = {k: cargar(v) for k, v in MODELOS.items()}
    idx = preds["CNN (mejor sola)"].index
    preds["CNN ensemble (3)"] = pd.Series(
        np.exp(np.mean([np.log(cargar(n).loc[idx]) for n in ENSEMBLE], axis=0)), index=idx)

    base = pd.DataFrame({"animal_id": man.loc[idx, "animal_id"], "peso_kg": man.loc[idx, "peso_kg"]})
    print(f"test: {len(base)} imágenes de {base.animal_id.nunique()} animales | "
          f"bootstrap de {a.repeticiones} repeticiones, remuestreando animales\n")

    filas = []
    for nombre, p in preds.items():
        d = base.assign(pred=p.loc[idx].to_numpy())
        media, lo, hi = bootstrap_por_animal(d, a.repeticiones)
        punt = metrics.mape(d.peso_kg, d.pred)
        filas.append({"modelo": nombre, "mape": punt, "ic95_desde": lo, "ic95_hasta": hi,
                      "amplitud": hi - lo})
    res = pd.DataFrame(filas).sort_values("mape")
    print(res.round(2).to_string(index=False))
    res.to_csv(config.RUNS / "intervalos_bootstrap.csv", index=False)

    mejor = res.iloc[0]
    con = res[res.modelo.str.startswith("(control)")]
    print(f"\nPara reportar: {mejor.modelo} = {mejor.mape:.1f}% "
          f"(IC 95%: {mejor.ic95_desde:.1f}-{mejor.ic95_hasta:.1f})")
    if len(con):
        c = con.iloc[0]
        solapan = not (mejor.ic95_hasta < c.ic95_desde or c.ic95_hasta < mejor.ic95_desde)
        print(f"Contra el control con sticker ({c.mape:.1f}%, IC {c.ic95_desde:.1f}-{c.ic95_hasta:.1f}): "
              f"los intervalos {'SE SUPERPONEN: la diferencia no es significativa' if solapan else 'NO se superponen'}.")
        print("  Es decir: con estos datos no se puede afirmar que el sticker mejore el resultado.")


if __name__ == "__main__":
    main()
