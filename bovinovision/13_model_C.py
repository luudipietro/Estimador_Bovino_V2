"""Etapa 10: Modelo C = distancias normalizadas por el sticker (36) + forma de la silueta + confianza de keypoints
+ sexo -> peso. Compara contra el Modelo B (sin marcador) y contra el sticker solo (baseline de la etapa 7).

Sobre medidas en cm reales (sec. 10 del plan del docx): el dataset AcmeAI NO documenta el diámetro físico del
sticker (revisado el PDF del proyecto), así que no se puede convertir de forma confiable a centímetros absolutos.
Se sigue trabajando en unidades relativas al sticker (ya usado en la etapa 7): es una escala real (el sticker mide
lo mismo en toda foto), solo que en una unidad arbitraria en vez de cm. No afecta al MAPE del modelo de peso.

Features nuevas respecto del Modelo B/sticker de la etapa 7:
  area_vaca / sticker_diam_px^2  : tamaño de la silueta en "unidades de sticker al cuadrado" (proxy de volumen 2D)
  bbox_alto, bbox_ancho (ídem, /diam sticker) : tamaño del cajón que encierra a la vaca
  conf_media, conf_min           : confianza media/mínima de los 9 keypoints (solo con features_pred; NaN con GT)
  sexo                           : F/M, one-hot

Guarda el modelo entrenado (runs/models/model_c.joblib) para reutilizarlo en 15_fusion.py y 17_predict.py.
Uso:  uv run python 13_model_C.py [--features features_pred]
"""
import argparse
import importlib

import joblib
import numpy as np
import pandas as pd

import config
from common import metrics

bl = importlib.import_module("09_baselines")


def armar_features(nombre_features: str, escala: str = "animal") -> pd.DataFrame:
    """escala='sticker': normaliza por el diámetro del marcador (requiere sticker en la foto).
    escala='animal'  : normaliza por el largo corporal del propio animal en píxeles (wither→pinbone).
                       Adimensional, NO necesita marcador: es el modo de producción (Modelo C')."""
    feat = pd.read_parquet(config.TABLAS / f"{nombre_features}.parquet").set_index("image_id")
    man = pd.read_parquet(config.TABLAS / "manifest.parquet").set_index("image_id")
    msk = pd.read_parquet(config.TABLAS / "mascaras.parquet").set_index("image_id")

    idx = feat.index
    d = msk.loc[idx, "sticker_diam_px"] if escala == "sticker" else feat["d_wither_pinbone"]
    df = feat.copy()
    # compacidad de la silueta: información que los ratios entre keypoints NO contienen
    df["c_area_vaca"] = msk.loc[idx, "area_vaca_px"] / d ** 2
    df["c_bbox_alto"] = (msk.loc[idx, "vaca_y1"] - msk.loc[idx, "vaca_y0"]) / d
    df["c_bbox_ancho"] = (msk.loc[idx, "vaca_x1"] - msk.loc[idx, "vaca_x0"]) / d
    df["c_sexo_M"] = (man.loc[idx, "sexo"] == "M").astype(float)

    kp_pred_path = config.TABLAS / f"{nombre_features.replace('features', 'keypoints')}.parquet"
    if kp_pred_path.exists() and nombre_features != "features_gt":
        kp = pd.read_parquet(kp_pred_path).set_index("image_id")
        cconf = [c for c in kp.columns if c.endswith("_conf")]
        df["c_conf_media"] = kp.loc[idx, cconf].mean(axis=1)
        df["c_conf_min"] = kp.loc[idx, cconf].min(axis=1)
    return df


def columnas_modelo_c(df: pd.DataFrame, escala: str = "animal") -> list[str]:
    """Columnas del Modelo C según la fuente de escala. Sin sticker usa los ratios adimensionales."""
    base = bl.cols(df, "s_") if escala == "sticker" else bl.cols(df, "r_")
    return base + [c for c in df.columns if c.startswith("c_")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_pred")
    ap.add_argument("--escala", choices=["animal", "sticker"], default="animal",
                    help="'animal' (default, sin marcador, modo de producción) | 'sticker' (histórico)")
    a = ap.parse_args()

    df = armar_features(a.features, a.escala)
    # sin sticker no hace falta descartar las imágenes cuyo marcador falta o está ocluido
    df = df.dropna(subset=bl.cols(df, "s_") if a.escala == "sticker" else bl.cols(df, "r_")).reset_index()
    print(f"[{a.features} | escala={a.escala}] {len(df)} imágenes")
    c_extra = [c for c in df.columns if c.startswith("c_")]
    print(f"features de forma/metadata: {c_extra}")

    tr = df[df.split == "train"]
    if a.escala == "sticker":
        conjuntos = {"B_ratios_solos": bl.cols(df, "r_"), "sticker_solo": bl.cols(df, "s_"),
                     "C_final": columnas_modelo_c(df, "sticker")}
    else:
        conjuntos = {"B_ratios_solos": bl.cols(df, "r_"),
                     "C'_ratios+forma": bl.cols(df, "r_") + [c for c in c_extra if c.startswith("c_area") or c.startswith("c_bbox")],
                     "C'_final": columnas_modelo_c(df, "animal")}
    filas = []
    for split in ("val", "test"):
        te = df[df.split == split]
        filas.append({"conjunto": "-", "modelo": "mediana", "split": split,
                      **metrics.reporte(tr.peso_kg, te.peso_kg, metrics.baseline_mediana(tr.peso_kg, te.peso_kg))})
        for nombre, c in conjuntos.items():
            for nom_m in ("ridge", "hgb"):
                p = bl.ajustar_predecir(nom_m, tr[c], tr.peso_kg, te[c])
                filas.append({"conjunto": nombre, "modelo": nom_m, "split": split,
                              **metrics.reporte(tr.peso_kg, te.peso_kg, p, nom_m)})
    res = pd.DataFrame(filas)[["split", "conjunto", "modelo", "n", "mape", "mape_mediana", "mejora_pct"]]
    res.to_csv(config.RUNS / f"model_c_{a.escala}.csv", index=False)
    print(res.round(2).to_string(index=False))

    # Modelo C final: se elige ridge o hgb por su MAPE en VALIDACIÓN (no en test) y se reentrena en train
    c_final = columnas_modelo_c(df, a.escala)
    clave = "C_final" if a.escala == "sticker" else "C'_final"
    val = res[(res.split == "val") & (res.conjunto == clave)]
    mejor = val.loc[val.mape.idxmin(), "modelo"]
    print(f"\nmodelo elegido por validación: {mejor}")
    modelo_final = bl.modelo(mejor).fit(tr[c_final], np.log(tr.peso_kg))
    (config.RUNS / "models").mkdir(exist_ok=True)
    joblib.dump({"modelo": modelo_final, "columnas": c_final, "features": a.features, "escala": a.escala,
                 "tipo": mejor}, config.RUNS / "models" / "model_c.joblib")

    te = df[df.split == "test"]
    p = np.exp(modelo_final.predict(te[c_final]))
    print(f"\nMAPE por banda (test, Modelo C, escala={a.escala}):")
    print(metrics.mape_por_banda(te.peso_kg, p).round(2).to_string(index=False))
    df.assign(pred_modelo_c=np.exp(modelo_final.predict(df[c_final])))[["image_id", "split", "peso_kg", "pred_modelo_c"]].to_parquet(
        config.TABLAS / "pred_modelo_c.parquet", index=False)


if __name__ == "__main__":
    main()
