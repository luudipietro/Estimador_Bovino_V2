"""Dataset propio, etapa 3a: mide la calidad de cada foto de vaca con los modelos del módulo `estimador/`.

Por cada foto de `dataset_propio/fotos.csv` corre el detector genérico (COCO 'cow') y el pipeline completo del
estimador (pose + CNN) y guarda señales objetivas: cuántas vacas se ven y cuál domina, si el animal toca el borde,
cuánto del cuadro ocupa, confianza de detección y de los 9 keypoints, relación ancho/alto de los keypoints (vista
lateral vs. frontal/trasera) y el peso que predice el estimador actual (solo para diagnóstico, NO es una evaluación:
ese modelo no extrapola y fue entrenado en Bangladesh).

Es la parte lenta (~1-3 s por foto en CPU), por eso guarda en data/propio/features_calidad.csv y retoma donde quedó.
Las decisiones (incluir / descartar) se toman en 42_propio_calidad.py, que es rápido y se puede reajustar.
Uso:  uv run python 42a_propio_features.py
"""
import sys
import time

import numpy as np
import pandas as pd
import pillow_heif
from PIL import Image, ImageOps

import config

pillow_heif.register_heif_opener()
sys.path.insert(0, str(config.ROOT.parent / "estimador"))
from estimador import CLASE_VACA_COCO, LADO_MAX, EstimadorPeso  # noqa: E402

SALIDA = config.DATA / "propio" / "features_calidad.csv"


def abrir_orientada(ruta):
    """Abre la foto con la orientación EXIF aplicada (el estimador no lo hace y los iPhone guardan fotos giradas)."""
    return ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")


def escena(est, im):
    """Cajas del detector sobre la imagen reducida a LADO_MAX: n vacas, dominancia, ocupación y contacto con el borde."""
    f = LADO_MAX / max(im.size)
    if f < 1:
        im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
    r = est.detector.predict(source=im, classes=[CLASE_VACA_COCO], conf=0.25, max_det=30, verbose=False)[0]
    if len(r.boxes) == 0:
        return {"n_vacas": 0}
    cajas = r.boxes.xyxy.cpu().numpy()
    areas = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
    o = np.argsort(-areas)
    x0, y0, x1, y1 = cajas[o[0]]
    margen = 0.01 * max(im.size)
    return {"n_vacas": len(cajas),
            "dominancia": float(areas[o[0]] / areas[o[1]]) if len(o) > 1 else float("inf"),
            "caja_ancho_frac": float((x1 - x0) / im.width), "caja_alto_frac": float((y1 - y0) / im.height),
            "caja_toca_borde": bool(x0 <= margen or y0 <= margen or x1 >= im.width - margen or y1 >= im.height - margen),
            "conf_caja": float(r.boxes.conf[o[0]])}


def main():
    fotos = pd.read_csv(config.PROPIO / "fotos.csv")
    fotos = fotos[fotos.uso == "candidata"]
    hechas = pd.read_csv(SALIDA) if SALIDA.exists() else pd.DataFrame(columns=["foto"])
    pend = fotos[~fotos.foto.isin(hechas.foto)]
    print(f"{len(fotos)} candidatas | ya medidas: {len(hechas)} | pendientes: {len(pend)}", flush=True)
    est = EstimadorPeso(config.ROOT.parent / "estimador" / "modelos")

    filas, t0 = [], time.time()
    for i, f in enumerate(pend.itertuples(), 1):
        try:
            im = abrir_orientada(config.PROPIO / f.foto)
            fila = {"foto": f.foto, **escena(est, im)}
            r = est.estimar(im)
            kp = r.get("keypoints")
            fila.update({"pred_peso_v1": r.get("peso_kg"), "conf_det": r.get("confianza_deteccion"),
                         "conf_kp_min": r.get("confianza_keypoint_minima"), "motivo_estimador": r.get("motivo")})
            if kp:
                a = np.array(list(kp.values()))
                fila["conf_kp_media"] = float(a[:, 2].mean())
                fila["aspecto_kp"] = float((a[:, 0].max() - a[:, 0].min()) / max(a[:, 1].max() - a[:, 1].min(), 1))
                # los keypoints vienen en coordenadas de la imagen reducida a LADO_MAX, no de la original
                ancho_red = im.width * min(1.0, LADO_MAX / max(im.size))
                fila["largo_kp_frac"] = float((a[:, 0].max() - a[:, 0].min()) / ancho_red)
            fila["ancho_px"], fila["alto_px"] = im.size
        except Exception as e:  # una foto rota no debe frenar 800
            fila = {"foto": f.foto, "error": f"{e.__class__.__name__}: {e}"}
        filas.append(fila)
        if i % 25 == 0 or i == len(pend):
            hechas = pd.concat([hechas, pd.DataFrame(filas)], ignore_index=True)
            hechas.to_csv(SALIDA, index=False)
            filas = []
            print(f"  {i}/{len(pend)}  ({(time.time() - t0) / i:.1f} s/foto)", flush=True)
    print(f"listo -> {SALIDA}")


if __name__ == "__main__":
    main()
