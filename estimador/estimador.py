"""Estimador de peso bovino a partir de una foto lateral. Módulo autocontenido, listo para integrar.

    from estimador import EstimadorPeso
    est = EstimadorPeso("modelos/")
    print(est.estimar("vaca.jpg"))
    # {'peso_kg': 187.4, 'confiable': True, 'motivo': None, 'confianza_deteccion': 0.94, ...}

Cómo funciona (dos modelos en cadena, ninguno necesita marcador ni sensores):
  1. YOLO-pose ubica 9 puntos anatómicos y con ellos se recorta al animal con un margen fijo.
  2. Una CNN (ConvNeXt-Tiny) mira ese recorte a 384x384 y predice el peso.

ESTADO DEL MODELO (importante): está entrenado con 4.540 fotos de bovinos de Bangladesh, de 36 a 621 kg, con el
78% entre 100 y 200 kg. Medimos que **no extrapola**: nunca predice por encima de ~270 kg. Sobre animales
argentinos de 400-500 kg va a devolver valores muy por debajo del real. Sirve HOY para integrar y probar el
flujo completo, no para dar pesos confiables en campo. Cuando exista el dataset propio se reentrena y se
reemplazan los pesos; la interfaz de este módulo no cambia.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Orden de los 9 keypoints tal como los produce el modelo de pose
KEYPOINTS = ["wither", "pinbone", "shoulderbone", "front_girth_top", "front_girth_bottom",
             "rear_girth_top", "rear_girth_bottom", "height_top", "height_bottom"]

LADO_MAX = 1024          # la imagen se reduce a este lado mayor antes de recortar (igual que en entrenamiento)
MARGEN_X, MARGEN_Y = 0.10, 0.15   # margen alrededor del animal al recortar
MEDIA = [0.485, 0.456, 0.406]
DESVIO = [0.229, 0.224, 0.225]

# Umbrales de la regla "no sé"
CONF_DETECCION_MIN = 0.35
CONF_KEYPOINT_MIN = 0.50
# Rango en el que el modelo fue entrenado; fuera de acá la predicción no es confiable
RANGO_ENTRENAMIENTO = (36.0, 621.0)


class EstimadorPeso:
    def __init__(self, carpeta_modelos: str | Path, dispositivo: str = "auto"):
        carpeta = Path(carpeta_modelos)
        cfg = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
        self.cfg = cfg
        self.tam = cfg.get("tam", 384)
        self.dispositivo = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if dispositivo == "auto" else dispositivo)

        from ultralytics import YOLO
        self.pose = YOLO(str(carpeta / cfg["archivo_pose"]))

        import timm
        self.cnn = timm.create_model(cfg["backbone"], pretrained=False, num_classes=1)
        self.cnn.load_state_dict(torch.load(carpeta / cfg["archivo_cnn"], map_location="cpu"))
        self.cnn.eval().to(self.dispositivo)

        from torchvision.transforms import v2
        self._v2 = v2
        self.tfm = v2.Compose([v2.Resize((self.tam, self.tam)),
                               v2.ToDtype(torch.float32, scale=True),
                               v2.Normalize(MEDIA, DESVIO)])

    # ------------------------------------------------------------------ interno
    def _abrir(self, imagen) -> Image.Image:
        if isinstance(imagen, Image.Image):
            return imagen.convert("RGB")
        if isinstance(imagen, (bytes, bytearray)):
            import io
            return Image.open(io.BytesIO(imagen)).convert("RGB")
        return Image.open(imagen).convert("RGB")

    def _recortar(self, im: Image.Image, xs, ys):
        mx, my = MARGEN_X * (xs.max() - xs.min()), MARGEN_Y * (ys.max() - ys.min())
        caja = (max(0, int(xs.min() - mx)), max(0, int(ys.min() - my)),
                min(im.width, int(xs.max() + mx)), min(im.height, int(ys.max() + my)))
        if caja[2] - caja[0] > 10 and caja[3] - caja[1] > 10:
            return im.crop(caja), caja
        return im, None

    # ------------------------------------------------------------------ público
    def estimar(self, imagen, tta: bool = True) -> dict:
        """imagen: ruta, bytes o PIL.Image. Devuelve un dict con el peso y la información de calidad."""
        im = self._abrir(imagen)
        ancho0, alto0 = im.size

        f = LADO_MAX / max(im.size)
        if f < 1:
            im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)

        res = self.pose.predict(source=im, imgsz=640, conf=0.05, max_det=3, verbose=False)[0]
        if len(res.boxes) == 0:
            return {"peso_kg": None, "confiable": False, "motivo": "no se detectó ningún bovino en la foto",
                    "resolucion": f"{ancho0}x{alto0}"}

        k = int(res.boxes.conf.argmax())
        conf_det = float(res.boxes.conf[k])
        xy = res.keypoints.xy[k].cpu().numpy()
        conf_kp = res.keypoints.conf[k].cpu().numpy()
        xs, ys = xy[:, 0], xy[:, 1]

        recorte, caja = self._recortar(im, xs, ys)
        x = self.tfm(self._v2.functional.pil_to_tensor(recorte)).unsqueeze(0).to(self.dispositivo)
        with torch.no_grad():
            log_peso = self.cnn(x).squeeze().item()
            if tta:  # promedia con la imagen espejada
                log_peso = (log_peso + self.cnn(torch.flip(x, dims=[3])).squeeze().item()) / 2
        peso = float(np.exp(log_peso))

        # --- reglas de "no sé" ---
        motivos = []
        if conf_det < CONF_DETECCION_MIN:
            motivos.append("el bovino no se distingue con claridad")
        if float(conf_kp.min()) < CONF_KEYPOINT_MIN:
            motivos.append("hay partes del cuerpo tapadas o fuera de cuadro")
        if caja is not None:
            margen = 0.01 * max(im.size)
            if caja[0] <= margen or caja[1] <= margen or caja[2] >= im.width - margen or caja[3] >= im.height - margen:
                motivos.append("el animal aparece cortado por el borde de la foto")
        if not (RANGO_ENTRENAMIENTO[0] <= peso <= RANGO_ENTRENAMIENTO[1]):
            motivos.append("el peso estimado cae fuera del rango con el que se entrenó el modelo")

        return {
            "peso_kg": round(peso, 1),
            "confiable": not motivos,
            "motivo": "; ".join(motivos) if motivos else None,
            "confianza_deteccion": round(conf_det, 3),
            "confianza_keypoint_minima": round(float(conf_kp.min()), 3),
            "keypoints": {n: [round(float(a), 1), round(float(b), 1), round(float(c), 3)]
                          for n, a, b, c in zip(KEYPOINTS, xs, ys, conf_kp)},
            "recorte": caja,
            "resolucion": f"{ancho0}x{alto0}",
            "modelo": self.cfg.get("version", "sin versión"),
        }

    def estimar_lote(self, imagenes, tta: bool = True) -> list[dict]:
        return [self.estimar(i, tta) for i in imagenes]
