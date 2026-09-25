"""Reporte final del rumbo SIN MARCADOR: consolida todos los modelos probados, por banda de peso, siempre
contra el baseline de la mediana. Es el entregable de números para el informe y la defensa.

Incluye el control con sticker para poder afirmar, con evidencia propia, cuánto costó sacarlo.
Uso:  uv run python 28_reporte_sin_sticker.py
"""
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from common import metrics

bl = importlib.import_module("09_baselines")
mc = importlib.import_module("13_model_C")


def main():
    man = pd.read_parquet(config.TABLAS / "manifest.parquet")
    y_tr = man[man.split == "train"].peso_kg

    # --- camino geométrico (sin marcador) ---
    feat = mc.armar_features("features_pred", "animal")
    cols_c = mc.columnas_modelo_c(feat, "animal")
    feat = feat.dropna(subset=cols_c).reset_index()
    tr_f, te_f = feat[feat.split == "train"], feat[feat.split == "test"]
    modelos = {
        "Mediana (no mira la foto)": metrics.baseline_mediana(y_tr, te_f.peso_kg),
        "B: ratios de keypoints": bl.ajustar_predecir("ridge", tr_f[bl.cols(feat, "r_")], tr_f.peso_kg,
                                                      te_f[bl.cols(feat, "r_")]),
        "C': ratios + forma + sexo": bl.ajustar_predecir("ridge", tr_f[cols_c], tr_f.peso_kg, te_f[cols_c]),
    }
    base = te_f.set_index("image_id")[["peso_kg"]]

    # --- camino visual ---
    r = pd.read_csv(config.RUNS / "cnn" / "resumen.csv")
    con = r[r.imagenes == "con_sticker"]

    def pred_de(nombre, split="test"):
        p = pd.read_parquet(config.TABLAS / f"pred_cnn_{nombre}.parquet").set_index("image_id")
        return p[p.split == split].pred_cnn

    # OJO: la columna mape_val del resumen NO es comparable entre corridas, porque las entrenadas con el dataset
    # unificado validan también sobre CID y Mendeley (más fáciles). Para elegir hay que medir todas sobre el
    # MISMO conjunto: el val de AcmeAI.
    ids_val = set(man[man.split == "val"].image_id)
    y_val = man.set_index("image_id").loc[sorted(ids_val), "peso_kg"]
    orden = []
    for n in r[r.imagenes == "sin_sticker"].nombre:
        if not (config.TABLAS / f"pred_cnn_{n}.parquet").exists():
            continue
        v = pred_de(n, "val")
        v = v[v.index.isin(ids_val)]
        orden.append({"nombre": n, "val_acmeai": metrics.mape(y_val.loc[v.index], v)})
    sin = pd.DataFrame(orden).sort_values("val_acmeai")
    print("corridas ordenadas por MAPE en el val de AcmeAI (criterio de selección):")
    print(sin.head(4).round(2).to_string(index=False))

    viejo = pd.read_parquet(config.TABLAS / "pred_modelo_a.parquet").set_index("image_id")
    modelos["A: CNN congelada (punto de partida)"] = viejo[viejo.split == "test"].pred_modelo_a.reindex(base.index)
    modelos["CNN fine-tuneada (mejor)"] = pred_de(sin.iloc[0].nombre).reindex(base.index)
    dfs = [pred_de(n).reindex(base.index) for n in sin.head(3).nombre]
    modelos["CNN ensemble (3 mejores)"] = np.exp(np.mean([np.log(d) for d in dfs], axis=0))
    if len(con):
        modelos["(control) CNN CON sticker"] = pred_de(con.iloc[0].nombre).reindex(base.index)

    y = base.peso_kg
    filas = [{"modelo": k, **metrics.reporte(y_tr, y, v, k)} for k, v in modelos.items()]
    res = pd.DataFrame(filas)[["modelo", "n", "mape", "mape_mediana", "mejora_pct"]]
    print("=== Resumen global (test, sin marcador salvo el control) ===")
    print(res.round(2).to_string(index=False))
    res.to_csv(config.RUNS / "reporte_sin_sticker_global.csv", index=False)

    banda = []
    for k, v in modelos.items():
        t = metrics.mape_por_banda(y, v)
        t.insert(0, "modelo", k)
        banda.append(t)
    banda = pd.concat(banda, ignore_index=True)
    tabla = banda.pivot(index="banda_kg", columns="modelo", values="mape")[list(modelos)]
    print("\n=== Por banda de peso (test) ===")
    print(tabla.round(1).to_string())
    banda.to_csv(config.RUNS / "reporte_sin_sticker_por_banda.csv", index=False)

    mejor = res[~res.modelo.str.startswith("(control)")].sort_values("mape").iloc[0]
    print(f"\nMejor SIN marcador: {mejor.modelo} -> {mejor.mape:.2f}% "
          f"(mejora {mejor.mejora_pct:.0f}% sobre la mediana)")
    if len(con):
        c = res[res.modelo.str.startswith("(control)")].iloc[0]
        print(f"Costo de sacar el marcador: {mejor.mape - c.mape:+.2f} puntos de MAPE "
              f"(con sticker {c.mape:.2f}%, sin sticker {mejor.mape:.2f}%)")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cols = [c for c in tabla.columns if not c.startswith("(control)")]
    x = np.arange(len(tabla.index))
    ancho = 0.8 / len(cols)
    for i, c in enumerate(cols):
        ax.bar(x + i * ancho, tabla[c], ancho, label=c)
    ax.axhline(15, color="red", ls="--", lw=1, label="objetivo 15%")
    ax.set_xticks(x + ancho * (len(cols) - 1) / 2)
    ax.set_xticklabels(tabla.index)
    ax.set_xlabel("banda de peso (kg)")
    ax.set_ylabel("MAPE (%)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(config.RUNS / "reporte_sin_sticker.png", dpi=120)
    print(f"-> {config.RUNS / 'reporte_sin_sticker.png'}")


if __name__ == "__main__":
    main()
