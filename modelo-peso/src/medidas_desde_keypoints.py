"""Del keypoint al kilo: distancias entre landmarks -> ratios -> peso.

Cierra el ciclo. Las distancias salen en PIXELES, pero no hace falta convertirlas
a centimetros: los RATIOS entre ellas son invariantes a la escala y ya cargan la
senal (5,53% de MAPE con medidas perfectas, contra 14,90% del baseline tonto).
Por eso este camino no necesita marcador, giroscopio ni LiDAR.

Uso (siempre parado en modelo-peso/):
  python src/medidas_desde_keypoints.py --pesos runs/pose7/weights/best.pt
  python src/medidas_desde_keypoints.py --labels data/pose/labels
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from unir import leer_medidas

CRUZ, PECHO_SUP, PECHO_INF, ENCUENTRO, LUMBAR, ISQUION, PEZUNA = range(7)


def distancias(kp):
    """kp: array (7,2) en pixeles. Devuelve las distancias que importan."""
    d = lambda a, b: float(np.linalg.norm(kp[a] - kp[b]))
    return {
        "px_altura_cruz": d(CRUZ, PEZUNA),
        "px_largo_corporal": d(ENCUENTRO, ISQUION),
        "px_prof_torax": d(PECHO_SUP, PECHO_INF),
        "px_altura_lumbar": d(LUMBAR, PEZUNA),
        "px_largo_lomo": d(CRUZ, LUMBAR),
        "px_grupa": d(LUMBAR, ISQUION),
    }


def a_ratios(d):
    """Invariantes a escala: es lo que permite prescindir de la calibracion metrica."""
    a, l, t = d["px_altura_cruz"], d["px_largo_corporal"], d["px_prof_torax"]
    lo, lu, gr = d["px_largo_lomo"], d["px_altura_lumbar"], d["px_grupa"]
    return {
        "r_largo_altura": l / a,
        "r_torax_altura": t / a,
        "r_torax_largo": t / l,
        "r_lomo_largo": lo / l,
        "r_lumbar_altura": lu / a,
        "r_grupa_largo": gr / l,
        "r_grupa_torax": gr / t,
    }


def desde_labels(carpeta: Path):
    """Lee anotaciones en formato YOLO pose: cls cx cy w h  (x y v)*7, normalizadas.

    Las coordenadas vienen normalizadas 0-1 (relativas al ancho/alto de cada
    imagen). Hay que volver a pixeles reales antes de calcular distancias: las
    imagenes son 1024x768 (no cuadradas), asi que normalizado != pixeles, y
    mezclar las dos escalas distorsiona distancias verticales vs horizontales
    de forma distinta (por eso desde_modelo() ya usa xy en vez de xyn).
    """
    from PIL import Image
    imgs_dir = Path(__file__).resolve().parents[1] / "etiquetado" / "imagenes"

    filas = []
    for p in sorted(carpeta.rglob("*.txt")):
        partes = p.read_text().split()
        if len(partes) < 5 + 7 * 3:
            print(f"  {p.name}: sin los 7 keypoints, salteado")
            continue
        ancho, alto = Image.open(imgs_dir / f"{p.stem}.jpg").size
        v = np.array(partes[5:5 + 21], float).reshape(7, 3)
        kp = v[:, :2] * np.array([ancho, alto])
        filas.append((int(p.stem), kp))
    return filas


def desde_modelo(pesos: str):
    from ultralytics import YOLO
    modelo = YOLO(pesos)
    imgs = sorted((Path(__file__).resolve().parents[1] / "etiquetado" / "imagenes").glob("*.jpg"))
    filas = []
    for p in imgs:
        r = modelo.predict(str(p), verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            print(f"  {p.name}: sin deteccion")
            continue
        filas.append((int(p.stem), r.keypoints.xy[0].cpu().numpy()))
    return filas


def evaluar(X, y, grupos, etiqueta):
    from sklearn.model_selection import GroupKFold, cross_val_predict
    cv = GroupKFold(n_splits=5)
    print(f"\n=== {etiqueta} ({X.shape[1]} features, {len(y)} animales) ===")
    for nombre, m in [
        ("Ridge", make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 40)))),
        ("Boosting", HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                   max_iter=400, min_samples_leaf=5,
                                                   l2_regularization=1.0, random_state=0)),
    ]:
        p = cross_val_predict(m, X, y, cv=cv, groups=grupos)
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"  {nombre:9s} MAPE {np.mean(np.abs((y-p)/y))*100:5.2f}%   "
              f"MAE {np.abs(y-p).mean():5.1f} kg   R2 {r2:+.3f}")


def main(args):
    kps = desde_labels(Path(args.labels)) if args.labels else desde_modelo(args.pesos)
    if not kps:
        print("No se obtuvo ningun keypoint.")
        return

    filas = []
    for aid, kp in kps:
        f = {"animal_id": aid}
        d = distancias(np.asarray(kp, float))
        f.update(d)
        f.update(a_ratios(d))
        filas.append(f)

    df = pd.DataFrame(filas).merge(leer_medidas()[["animal_id", "peso_kg"]], on="animal_id")
    y = df["peso_kg"].to_numpy(float)
    g = df["animal_id"].to_numpy()
    print(f"\n{len(df)} animales con keypoints y peso")
    print(f"Baseline tonto (mediana): {np.mean(np.abs((y-np.median(y))/y))*100:.2f}%")

    cols_r = [c for c in df.columns if c.startswith("r_")]
    cols_px = [c for c in df.columns if c.startswith("px_")]
    evaluar(df[cols_r].to_numpy(float), y, g, "Solo ratios (SIN escala metrica)")
    evaluar(df[cols_r + cols_px].to_numpy(float), y, g, "Ratios + distancias en px")

    df.to_csv(config.TABLAS / "keypoints.csv", index=False)
    print(f"\n-> {config.TABLAS / 'keypoints.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default="runs/pose7/weights/best.pt")
    ap.add_argument("--labels", help="carpeta con .txt en formato YOLO pose")
    main(ap.parse_args())
