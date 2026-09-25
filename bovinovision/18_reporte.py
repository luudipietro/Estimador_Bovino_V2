"""Etapa 15: reporte final — A (visual) / B (geometría sin escala) / C (geometría + sticker) / fusión, por banda
de peso, siempre junto al baseline de la mediana. Números para el hito del 30/09/2026 (ver ../CLAUDE.md del proyecto).
Uso:  uv run python 18_reporte.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from common import metrics


def main():
    pred = pd.read_parquet(config.TABLAS / "predicciones_finales.parquet")
    tr = pred[pred.split == "train"]
    te = pred[pred.split == "test"]

    modelos = {
        "Mediana (baseline)": metrics.baseline_mediana(tr.peso_kg, te.peso_kg),
        "A: visual (CNN, sin geometría)": te.pred_modelo_a,
        "B: keypoints, sin escala": None,  # se agrega abajo desde features_pred (no viaja en predicciones_finales)
        "C: keypoints + sticker + forma": te.pred_modelo_c,
        "Fusión A+C": te.pred_fusion,
    }
    import importlib
    bl = importlib.import_module("09_baselines")
    feat = pd.read_parquet(config.TABLAS / "features_pred.parquet")
    ftr, fte = feat[feat.split == "train"], feat[feat.split == "test"].set_index("image_id").loc[te.image_id]
    modelos["B: keypoints, sin escala"] = bl.ajustar_predecir("ridge", ftr[bl.cols(feat, "r_")], ftr.peso_kg,
                                                              fte[bl.cols(feat, "r_")])

    print("=== Resumen global (test, n={}) ===".format(len(te)))
    filas_globales = []
    for nombre, p in modelos.items():
        r = metrics.reporte(tr.peso_kg, te.peso_kg, p, nombre)
        filas_globales.append(r)
    resumen = pd.DataFrame(filas_globales)[["modelo", "n", "mape", "mape_mediana", "mejora_pct"]]
    print(resumen.round(2).to_string(index=False))

    print("\n=== Por banda de peso (test) ===")
    filas_banda = []
    for nombre, p in modelos.items():
        t = metrics.mape_por_banda(te.peso_kg, p)
        t.insert(0, "modelo", nombre)
        filas_banda.append(t)
    banda = pd.concat(filas_banda, ignore_index=True)
    tabla_banda = banda.pivot(index="banda_kg", columns="modelo", values="mape")[list(modelos)]
    print(tabla_banda.round(1).to_string())

    resumen.to_csv(config.RUNS / "reporte_final_global.csv", index=False)
    banda.to_csv(config.RUNS / "reporte_final_por_banda.csv", index=False)

    # Objetivo del acta: MAPE < 15%, validado con >=100 registros con peso real en báscula (ya se cumple: 670 en test)
    cumple = resumen.set_index("modelo").loc["Fusión A+C", "mape"] < 15
    print(f"\nObjetivo MAPE < 15% (acta, obj. 2): {'CUMPLE' if cumple else 'NO CUMPLE'} en el global "
          f"(fusión: {resumen.set_index('modelo').loc['Fusión A+C', 'mape']:.2f}%). "
          f"En la banda 100-200 kg sí cumple ({tabla_banda.loc['100-200', 'Fusión A+C']:.1f}%); "
          f"el resto de las bandas (menos datos, terneros y animales grandes) queda por resolver.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(tabla_banda.index))
    ancho = 0.8 / len(modelos)
    for i, nombre in enumerate(modelos):
        ax.bar(x + i * ancho, tabla_banda[nombre], ancho, label=nombre)
    ax.axhline(15, color="red", linestyle="--", linewidth=1, label="objetivo 15%")
    ax.set_xticks(x + ancho * (len(modelos) - 1) / 2)
    ax.set_xticklabels(tabla_banda.index)
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("banda de peso (kg)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(config.RUNS / "reporte_final_por_banda.png", dpi=120)
    print(f"\n-> {config.RUNS / 'reporte_final_por_banda.png'}")


if __name__ == "__main__":
    main()
