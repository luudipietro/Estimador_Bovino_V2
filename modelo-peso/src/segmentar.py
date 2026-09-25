"""Segmenta el bovino en cada imagen con YOLO-seg preentrenado en COCO.

La clase 'cow' ya viene en COCO, asi que esto corre sin entrenar nada.
Guarda una mascara binaria por imagen y un informe de que fallo.

Uso:  python src/segmentar.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# COCO: 19=cow, 18=sheep, 17=horse. El ganado a veces se detecta como caballo.
CLASES_ANIMAL = {19: "cow", 17: "horse", 18: "sheep", 20: "elephant"}
ANCHO_TRABAJO = 1024


def cargar_reducida(ruta: Path):
    img = cv2.imread(str(ruta))
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > ANCHO_TRABAJO:
        escala = ANCHO_TRABAJO / w
        img = cv2.resize(img, (ANCHO_TRABAJO, int(h * escala)), interpolation=cv2.INTER_AREA)
    return img


def mascara_principal(resultado, forma):
    if resultado.masks is None or len(resultado.masks) == 0:
        return None, None
    cajas = resultado.boxes
    candidatos = []
    for i in range(len(cajas)):
        cls = int(cajas.cls[i])
        if cls not in CLASES_ANIMAL:
            continue
        m = resultado.masks.data[i].cpu().numpy()
        m = cv2.resize(m, (forma[1], forma[0]), interpolation=cv2.INTER_NEAREST)
        # el bovino en foco es el de mayor area; prioriza 'cow' sobre el resto
        candidatos.append((cls == 19, float(m.sum()), cls, m))
    if not candidatos:
        return None, None
    candidatos.sort(key=lambda c: (c[0], c[1]), reverse=True)
    es_cow, area, cls, m = candidatos[0]
    return (m > 0.5).astype(np.uint8) * 255, CLASES_ANIMAL[cls]


def identificar(ruta: Path):
    """side view/12.png -> ('side', '12'). El nombre del archivo es el id del animal."""
    vista = "back" if "back" in ruta.parent.name.lower() else "side"
    return vista, ruta.stem


def main():
    imgs = sorted(
        (p for p in config.CRUDO.rglob("*")
         if p.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        key=lambda p: (p.parent.name, int(p.stem) if p.stem.isdigit() else 0),
    )
    if not imgs:
        print(f"No hay imagenes en {config.CRUDO}. Corre primero src/preparar.py")
        return

    print(f"{len(imgs)} imagenes. Cargando modelo...")
    modelo = YOLO("yolo11n-seg.pt")

    filas = []
    for i, ruta in enumerate(imgs, 1):
        vista, animal = identificar(ruta)
        clave = f"{vista}_{animal}"

        img = cargar_reducida(ruta)
        if img is None:
            filas.append({"clave": clave, "ok": False, "motivo": "no se pudo leer"})
            continue

        res = modelo.predict(img, verbose=False, conf=0.25)[0]
        mask, clase = mascara_principal(res, img.shape[:2])

        if mask is None:
            filas.append({"clave": clave, "vista": vista, "animal_id": animal,
                          "ok": False, "motivo": "sin deteccion de animal"})
        else:
            cv2.imwrite(str(config.MASCARAS / f"{clave}.png"), mask)
            filas.append({
                "clave": clave,
                "vista": vista,
                "animal_id": animal,
                "ok": True,
                "clase_detectada": clase,
                "area_px": int((mask > 0).sum()),
                "cobertura": float((mask > 0).mean()),
            })

        if i % 20 == 0 or i == len(imgs):
            print(f"  {i}/{len(imgs)}")

    df = pd.DataFrame(filas)
    df.to_csv(config.TABLAS / "segmentacion.csv", index=False)

    ok = int(df["ok"].sum())
    print(f"\nSegmentadas {ok}/{len(df)} ({ok/len(df)*100:.1f}%)")
    print("\nPor vista:")
    print(df.groupby("vista")["ok"].agg(["sum", "count"]).to_string())
    if (~df["ok"]).any():
        print("\nFallos:")
        print(df[~df["ok"]][["clave", "motivo"]].to_string(index=False))
    print("\nClase detectada:")
    print(df[df["ok"]]["clase_detectada"].value_counts().to_string())


if __name__ == "__main__":
    main()
