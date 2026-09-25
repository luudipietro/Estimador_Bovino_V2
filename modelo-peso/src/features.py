"""Descriptores de forma INVARIANTES A ESCALA desde las mascaras.

Todo se normaliza por el bounding box del animal: no hay ni un centimetro
en este archivo, a proposito. La informacion de peso sale de la FORMA
(alometria) y no del tamano aparente, que depende de la distancia.

Uso:  python src/features.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

N_COLUMNAS = 16  # resolucion del perfil de silueta


def limpiar(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return None
    mayor = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (etiquetas == mayor).astype(np.uint8)


def perfiles(binaria):
    """Linea superior, linea inferior y espesor, en N columnas normalizadas."""
    ys, xs = np.nonzero(binaria)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    ancho, alto = x1 - x0 + 1, y1 - y0 + 1
    recorte = binaria[y0:y1 + 1, x0:x1 + 1]

    superior, inferior, espesor = [], [], []
    bordes = np.linspace(0, ancho, N_COLUMNAS + 1).astype(int)
    for i in range(N_COLUMNAS):
        banda = recorte[:, bordes[i]:max(bordes[i + 1], bordes[i] + 1)]
        filas = np.nonzero(banda.any(axis=1))[0]
        if len(filas) == 0:
            superior.append(1.0); inferior.append(0.0); espesor.append(0.0)
        else:
            superior.append(filas.min() / alto)
            inferior.append(filas.max() / alto)
            espesor.append((filas.max() - filas.min() + 1) / alto)
    return np.array(superior), np.array(inferior), np.array(espesor), recorte, ancho, alto


def extraer(mask, espejar=False):
    """Un animal puede estar mirando a izquierda o derecha, y ninguna heuristica
    de orientacion resulto confiable (en un toro pesado el tren delantero es mas
    grueso que el trasero). En vez de adivinar, emitimos las dos orientaciones y
    el modelo aprende la invarianza; de paso duplica las muestras."""
    binaria = limpiar(mask)
    if binaria is None or binaria.sum() < 500:
        return None
    if espejar:
        binaria = binaria[:, ::-1].copy()
    superior, inferior, espesor, recorte, ancho, alto = perfiles(binaria)

    area = float(binaria.sum())
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contornos, key=cv2.contourArea)
    perimetro = cv2.arcLength(c, True)
    casco = cv2.convexHull(c)
    area_casco = cv2.contourArea(casco)
    (_, _), (w_rot, h_rot), _ = cv2.minAreaRect(c)
    lado_may, lado_men = max(w_rot, h_rot), max(min(w_rot, h_rot), 1e-6)

    hu = cv2.HuMoments(cv2.moments(binaria)).flatten()
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    m = cv2.moments(binaria)
    cx = m["m10"] / m["m00"] / binaria.shape[1]
    cy = m["m01"] / m["m00"] / binaria.shape[0]

    tercio = recorte.shape[1] // 3
    a_del = recorte[:, :tercio].sum() / area
    a_med = recorte[:, tercio:2 * tercio].sum() / area
    a_tra = recorte[:, 2 * tercio:].sum() / area

    f = {
        # Tamano aparente en pixeles: DEPENDE de la distancia de captura.
        # Se guarda aparte para poder medir cuanto aporta cuando el protocolo
        # de captura fija la distancia (en este dataset fue 1 m).
        "px_area": area,
        "px_ancho": float(ancho),
        "px_alto": float(alto),
        "px_diagonal": float(np.hypot(ancho, alto)),

        "relacion_aspecto": ancho / alto,
        "extent": area / (ancho * alto),
        "solidez": area / max(area_casco, 1e-6),
        "elongacion": lado_may / lado_men,
        "circularidad": 4 * np.pi * area / max(perimetro ** 2, 1e-6),
        "perim_norm": perimetro / max(2 * (ancho + alto), 1e-6),
        "centroide_x": cx,
        "centroide_y": cy,
        "area_delantera": a_del,
        "area_media": a_med,
        "area_trasera": a_tra,
        # el espesor maximo relativo es el proxy escala-libre de la profundidad toracica
        "espesor_max": float(espesor.max()),
        "espesor_medio": float(espesor.mean()),
        "espesor_std": float(espesor.std()),
        "topline_pendiente": float(np.polyfit(np.arange(N_COLUMNAS), superior, 1)[0]),
        "topline_std": float(superior.std()),
    }
    for i in range(N_COLUMNAS):
        f[f"sup_{i:02d}"] = float(superior[i])
        f[f"inf_{i:02d}"] = float(inferior[i])
        f[f"esp_{i:02d}"] = float(espesor[i])
    for i, v in enumerate(hu):
        f[f"hu_{i+1}"] = float(v)
    return f


def main():
    mascaras = sorted(config.MASCARAS.glob("*.png"))
    if not mascaras:
        print(f"No hay mascaras en {config.MASCARAS}. Corre primero src/segmentar.py")
        return

    filas = []
    for p in mascaras:
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        for espejar in (False, True):
            f = extraer(mask, espejar=espejar)
            if f is None:
                if not espejar:
                    print(f"  descartada (mascara degenerada): {p.name}")
                break
            f["imagen"] = p.stem
            f["espejada"] = int(espejar)
            filas.append(f)

    df = pd.DataFrame(filas)
    salida = config.TABLAS / "features.csv"
    df.to_csv(salida, index=False)
    n_desc = len([c for c in df.columns if c not in ("imagen", "espejada")])
    print(f"\n{df['imagen'].nunique()} siluetas x 2 orientaciones = {len(df)} filas")
    print(f"{n_desc} descriptores, todos invariantes a escala")
    print(f"Guardado en {salida}")


if __name__ == "__main__":
    main()
