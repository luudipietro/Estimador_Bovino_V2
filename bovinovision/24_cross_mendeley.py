"""Fase 4a: ¿el sistema entrenado en Bangladesh sirve en otro país, otra raza y otro rango de peso?

Se aplica tal cual a Mendeley (72 bovinos de Mongolia Interior, 341-644 kg, iPhone 13): primero el YOLO-pose
entrenado con AcmeAI para ubicar los 9 keypoints, después el modelo de regresión geométrica (ratios, sin
marcador) entrenado también con AcmeAI. Nada se reentrena: es transferencia directa.

Se espera que dé mal, y el punto es medir CUÁNTO: AcmeAI casi no tiene animales en ese rango (13 imágenes sobre
341 kg), así que esto es extrapolación, no solo cambio de dominio. El número sirve como evidencia para el informe
y para dimensionar el dataset propio argentino (Fase 4b).

Uso:  uv run python 24_cross_mendeley.py
"""
import importlib

import numpy as np
import pandas as pd

import config
from common import features, metrics
from common.acmeai import KP_CANONICOS

bl = importlib.import_module("09_baselines")


def keypoints_mendeley(pesos, imgsz=640):
    from ultralytics import YOLO
    man = pd.read_parquet(config.TABLAS / "manifest_mendeley.parquet")
    modelo = YOLO(pesos)
    filas = []
    for r in man.itertuples():
        res = modelo.predict(source=r.ruta_imagen, imgsz=imgsz, conf=0.05, max_det=3, verbose=False)[0]
        fila = {"image_id": r.image_id, "n_det": len(res.boxes)}
        if len(res.boxes):
            k = int(res.boxes.conf.argmax())
            xy, kc = res.keypoints.xy[k].cpu().numpy(), res.keypoints.conf[k].cpu().numpy()
            fila["det_conf"] = float(res.boxes.conf[k])
            for j, n in enumerate(KP_CANONICOS):
                fila[f"{n}_x"], fila[f"{n}_y"], fila[f"{n}_conf"] = float(xy[j, 0]), float(xy[j, 1]), float(kc[j])
        filas.append(fila)
    return pd.DataFrame(filas).set_index("image_id"), man


def main():
    pesos = config.RUNS / "pose" / "pose9" / "weights" / "best.pt"
    kp, man = keypoints_mendeley(pesos)
    print(f"Mendeley: {len(kp)} imágenes | sin detección: {kp.det_conf.isna().sum()} | "
          f"confianza media de detección: {kp.det_conf.mean():.3f}")
    kp = kp.dropna(subset=["det_conf"])
    man = man.set_index("image_id").loc[kp.index]

    # features geométricas adimensionales (sin marcador), igual que el Modelo B/C' en AcmeAI
    d = features.distancias_px(kp)
    r_mend = features.ratios(d)

    acme = pd.read_parquet(config.TABLAS / "features_pred.parquet")
    cols = bl.cols(acme, "r_")
    tr = acme[acme.split == "train"]

    print("\n--- Transferencia directa (entrenado en AcmeAI, evaluado en Mendeley) ---")
    for nom in ("ridge", "hgb"):
        p = bl.ajustar_predecir(nom, tr[cols], tr.peso_kg, r_mend[cols])
        print(f"{nom:6s} MAPE {metrics.mape(man.peso_kg, p):6.2f}% | MAE {metrics.mae(man.peso_kg, p):6.1f} kg | "
              f"predicho {p.min():.0f}-{p.max():.0f} kg (real {man.peso_kg.min():.0f}-{man.peso_kg.max():.0f})")
    print(f"mediana de AcmeAI aplicada a Mendeley: "
          f"MAPE {metrics.mape(man.peso_kg, metrics.baseline_mediana(tr.peso_kg, man.peso_kg)):.2f}%")
    print(f"mediana de la PROPIA Mendeley (techo trivial si tuviéramos datos locales): "
          f"MAPE {metrics.mape(man.peso_kg, np.full(len(man), man.peso_kg.median())):.2f}%")

    # ¿los ratios (la forma del animal) al menos correlacionan con el peso en el dominio nuevo?
    corr = pd.Series({c: np.corrcoef(r_mend[c], man.peso_kg)[0, 1] for c in cols}).abs().sort_values(ascending=False)
    print(f"\ncorrelación |r| de los ratios con el peso dentro de Mendeley: "
          f"máx {corr.iloc[0]:.2f} ({corr.index[0]}), mediana {corr.median():.2f}")

    salida = r_mend.assign(peso_kg=man.peso_kg.values, animal_id=man.animal_id.values, split=man.split.values)
    salida.reset_index().to_parquet(config.TABLAS / "features_mendeley.parquet", index=False)
    print(f"-> {config.TABLAS / 'features_mendeley.parquet'}")


if __name__ == "__main__":
    main()
