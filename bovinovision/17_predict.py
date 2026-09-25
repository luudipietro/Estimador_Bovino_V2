"""Etapa 14: pipeline completo imagen -> peso + confianza (sec. 17 del plan: flujo de uso propuesto).

Encadena todo lo que ya se entrenó/guardó: YOLO-pose (runs/pose/pose9), Modelo C (runs/models/model_c.joblib),
Modelo A -embedding resnet18 + ridge- (runs/models/model_a.joblib) y el stacker de fusión
(runs/models/fusion_stacker.joblib). No reentrena nada: solo corre inferencia sobre UNA imagen.

Requiere la máscara semántica (fondo/sticker/vaca) de la imagen para calcular la escala real y las features de
forma: en producción esa máscara la daría el segmentador (etapa 4, aún no entrenado en este repo — pendiente,
ver 05_mascaras.py que hoy solo lee máscaras ya provistas por el dataset). Por eso este script, por ahora, se usa
sobre imágenes del propio dataset AcmeAI (pasa el image_id, no una foto nueva); es el "cableado" a completar
cuando exista un segmentador propio.

Uso:  uv run python 17_predict.py --image-id B3/100_s_181_F.jpg
"""
import argparse

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
from common import features, masks
from common.acmeai import KP_CANONICOS

UMBRAL_CONF_MIN = 0.5


def cargar_todo():
    from ultralytics import YOLO
    pose = YOLO(config.RUNS / "pose" / "pose9" / "weights" / "best.pt")
    model_c = joblib.load(config.RUNS / "models" / "model_c.joblib")
    model_a = joblib.load(config.RUNS / "models" / "model_a.joblib")
    stacker = joblib.load(config.RUNS / "models" / "fusion_stacker.joblib")
    return pose, model_c, model_a, stacker


def predecir(image_id: str, pose, model_c, model_a, stacker) -> dict:
    man = pd.read_parquet(config.TABLAS / "manifest.parquet").set_index("image_id").loc[image_id]
    im = Image.open(config.ACMEAI / man.ruta_imagen).convert("RGB")

    # 1) keypoints (YOLO-pose)
    res = pose.predict(source=str(config.ACMEAI / man.ruta_imagen), imgsz=640, conf=0.05, max_det=3, verbose=False)[0]
    if len(res.boxes) == 0:
        return {"peso_kg": None, "motivo_rechazo": "no se detectó el bovino"}
    k = int(res.boxes.conf.argmax())
    xyn, kc = res.keypoints.xyn[k].numpy(), res.keypoints.conf[k].numpy()
    kp_row = {}
    for j, nombre in enumerate(KP_CANONICOS):
        kp_row[f"{nombre}_x"], kp_row[f"{nombre}_y"] = xyn[j, 0] * man.ancho, xyn[j, 1] * man.alto
    kp = pd.DataFrame([kp_row])
    det_conf, conf_min, conf_media = float(res.boxes.conf[k]), float(kc.min()), float(kc.mean())

    # 2) máscara -> sticker + silueta (ver limitación arriba: hoy viene del dataset, no de un segmentador propio)
    rgba = np.array(Image.open(config.ACMEAI / man.ruta_mascara).convert("RGBA"))
    vaca, sticker_m = masks.separar(rgba)
    est = masks.estadisticas(vaca, sticker_m)
    sticker_ok = est.get("n_comp_sticker") == 1 and est.get("sticker_redondez", 0) >= 0.9
    if "sticker_diam_px" not in est:
        return {"peso_kg": None, "motivo_rechazo": "no se encontró el marcador de referencia en la imagen"}

    # 3) features geométricas normalizadas por el sticker + forma + sexo (Modelo C)
    d = features.distancias_px(kp)
    s = features.normalizadas_por_sticker(d, pd.Series([est["sticker_diam_px"]]))
    feat_c = s.copy()
    feat_c["c_area_vaca"] = est["area_vaca_px"] / est["sticker_diam_px"] ** 2
    feat_c["c_bbox_alto"] = (est["vaca_y1"] - est["vaca_y0"]) / est["sticker_diam_px"]
    feat_c["c_bbox_ancho"] = (est["vaca_x1"] - est["vaca_x0"]) / est["sticker_diam_px"]
    feat_c["c_sexo_M"] = float(man.sexo == "M")
    feat_c["c_conf_media"], feat_c["c_conf_min"] = conf_media, conf_min
    pred_c = float(np.exp(model_c["modelo"].predict(feat_c[model_c["columnas"]])[0]))

    # 4) embedding visual (Modelo A)
    import timm
    from torchvision import transforms
    backbone = timm.create_model(model_a["backbone"], pretrained=True, num_classes=0, global_pool="avg").eval()
    cfg = timm.data.resolve_data_config({}, model=backbone)
    tfm = transforms.Compose([transforms.Resize((cfg["input_size"][1], cfg["input_size"][2])), transforms.ToTensor(),
                              transforms.Normalize(cfg["mean"], cfg["std"])])
    with torch.no_grad():
        emb = backbone(tfm(im).unsqueeze(0)).numpy()
    feat_a = pd.DataFrame(emb, columns=model_a["columnas"])
    pred_a = float(np.exp(model_a["modelo"].predict(feat_a)[0]))

    # 5) fusión
    x_stack = pd.DataFrame({"log_a": [np.log(pred_a)], "log_c": [np.log(pred_c)]})
    pred_fusion = float(np.exp(stacker.predict(x_stack)[0]))

    aceptar = sticker_ok and conf_min >= UMBRAL_CONF_MIN
    return {"peso_kg": round(pred_fusion, 1), "peso_kg_visual": round(pred_a, 1), "peso_kg_geom": round(pred_c, 1),
            "confiable": aceptar, "det_conf": round(det_conf, 3), "conf_min_keypoints": round(conf_min, 3),
            "sticker_ok": sticker_ok, "motivo_rechazo": None if aceptar else "calidad de imagen insuficiente"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-id", required=True, help='ej. "B3/100_s_181_F.jpg" (ver manifest.parquet)')
    a = ap.parse_args()
    pose, model_c, model_a, stacker = cargar_todo()
    r = predecir(a.image_id, pose, model_c, model_a, stacker)
    man = pd.read_parquet(config.TABLAS / "manifest.parquet").set_index("image_id").loc[a.image_id]
    print(f"peso real (báscula): {man.peso_kg} kg")
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
