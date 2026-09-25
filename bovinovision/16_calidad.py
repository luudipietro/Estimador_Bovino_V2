"""Etapa 13: score de calidad y regla "no sé" (sec. 13 del plan: mejor rechazar una imagen mala que dar un
peso aparentemente preciso pero incorrecto).

Señales disponibles (todas ya calculadas por etapas anteriores, sin volver a correr el modelo de pose/depth):
  conf_media, conf_min   : confianza media/mínima de los 9 keypoints predichos por YOLO-pose
  det_conf                : confianza de la detección del bovino
  sticker_ok               : sticker con 1 sola componente y redondez >= 0.9 (no ocluido/cortado)
Score = combinación simple (0-1) de esas señales; se ordenan las imágenes de test por score y se mide cómo baja
el MAPE de la fusión (15_fusion.py) al ir rechazando la cola de peor calidad -> curva MAPE vs cobertura.
Uso:  uv run python 16_calidad.py
"""
import numpy as np
import pandas as pd

import config
from common import metrics


def main():
    pred = pd.read_parquet(config.TABLAS / "predicciones_finales.parquet").set_index("image_id")
    kp = pd.read_parquet(config.TABLAS / "keypoints_pred.parquet").set_index("image_id")
    msk = pd.read_parquet(config.TABLAS / "mascaras.parquet").set_index("image_id")

    idx = pred.index
    cconf = [c for c in kp.columns if c.endswith("_conf")]
    df = pd.DataFrame({
        "conf_media": kp.loc[idx, cconf].mean(axis=1),
        "conf_min": kp.loc[idx, cconf].min(axis=1),
        "det_conf": kp.loc[idx, "det_conf"],
        "sticker_ok": ((msk.loc[idx, "n_comp_sticker"] == 1) & (msk.loc[idx, "sticker_redondez"] >= 0.9)).astype(float),
    }, index=idx)
    # score simple 0-1: promedio de las 3 señales continuas, penalizado a la mitad si el sticker está sucio
    # (el sticker solo importa para la escala; una imagen puede tener buena pose y mal sticker, o viceversa)
    df["score"] = (df.conf_media + df.conf_min + df.det_conf) / 3 * np.where(df.sticker_ok == 1, 1.0, 0.5)

    df = df.join(pred[["split", "peso_kg", "pred_fusion"]])
    te = df[df.split == "test"].sort_values("score")
    print(f"Distribución del score (test): min {te.score.min():.3f} | mediana {te.score.median():.3f} | "
          f"max {te.score.max():.3f} | sticker_ok: {te.sticker_ok.mean():.1%}")

    filas = []
    for rechazo in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        n_rechazadas = int(round(len(te) * rechazo))
        acept = te.iloc[n_rechazadas:]
        filas.append({"rechazo_pct": int(rechazo * 100), "n_aceptadas": len(acept),
                      "umbral_score": te.score.iloc[n_rechazadas] if n_rechazadas else float("-inf"),
                      "mape": metrics.mape(acept.peso_kg, acept.pred_fusion)})
    curva = pd.DataFrame(filas)
    curva.to_csv(config.RUNS / "calidad_mape_vs_cobertura.csv", index=False)
    print("\nMAPE (test, fusión) según % de imágenes de peor calidad rechazadas ('no sé'):")
    print(curva.round(3).to_string(index=False))

    # Regla concreta propuesta: rechazar si el sticker no es válido O si la confianza mínima de keypoints es baja
    UMBRAL_CONF_MIN = 0.5
    regla = (df.sticker_ok == 1) & (df.conf_min >= UMBRAL_CONF_MIN)
    te_r = df[(df.split == "test")]
    aceptadas, rechazadas = te_r[regla[te_r.index]], te_r[~regla[te_r.index]]
    print(f"\nRegla propuesta (sticker_ok AND conf_min>={UMBRAL_CONF_MIN}): "
          f"acepta {len(aceptadas)}/{len(te_r)} ({len(aceptadas) / len(te_r):.1%}) en test")
    print(f"  MAPE aceptadas: {metrics.mape(aceptadas.peso_kg, aceptadas.pred_fusion):.2f}% | "
          f"MAPE rechazadas (si igual se estimaran): {metrics.mape(rechazadas.peso_kg, rechazadas.pred_fusion):.2f}%")


if __name__ == "__main__":
    main()
