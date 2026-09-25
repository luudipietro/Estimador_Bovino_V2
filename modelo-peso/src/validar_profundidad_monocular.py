"""Alternativa SIN marcador: recuperar la escala con profundidad monocular metrica.

Que pregunta contesta
---------------------
`validar_metodo_acmeai.py` concluyo que hace falta pegarle una calcomania al
animal porque las distancias en pixeles no tienen escala. Este script prueba la
hipotesis competidora: que un modelo de profundidad monocular metrica (Depth Pro,
UniDepth, Depth Anything V2 metric) recupera esa escala solo, sin tocar al animal
-- y ademas corrige algo que el sticker NO puede corregir: que cada landmark esta
a una profundidad distinta de la camara (el escorzo por yaw/perspectiva, el techo
estructural que quedo documentado en README.md).

Corre sobre el MISMO dataset de AcmeAI que ya se uso, asi que es un A/B directo
contra el 15,8% de MAPE del metodo con sticker, sin recolectar una sola foto nueva.

Lo que hace, por imagen
-----------------------
1. Profundidad metrica Z(u,v) en metros + distancia focal en pixeles.
2. Retroproyeccion de los 9 keypoints a 3D:  X=(u-cx)Z/f, Y=(v-cy)Z/f, Z.
   Z se muestrea con la mediana de los pixeles DE ADENTRO de la mascara del
   animal en un disco chico: varios landmarks caen justo sobre la silueta, donde
   la profundidad salta al fondo y un muestreo puntual da basura.
3. Las 36 distancias entre pares, ahora en METROS REALES (no en unidades de
   sticker, no en pixeles).
4. Dos features metricas que salen de la mascara y que ninguna distancia captura:
   area proyectada en m2, y un volumen por solido de revolucion en m3.

Y el diagnostico decisivo, que no necesita entrenar nada
--------------------------------------------------------
El sticker esta en TODAS las fotos del dataset. Se lo usa como REGLA DE
VALIDACION (nunca como entrada del modelo): si la escala que deduce la red de
profundidad correlaciona fuerte con la que da el sticker, el sticker sobra. Es un
scatter plot y un coeficiente, contra 4.500 fotos con peso de bascula.

Requisitos
----------
    uv add transformers timm
El modelo por defecto (apple/DepthPro-hf) es pesado en CPU (~20-40 s/imagen).
Para una primera pasada usar --limite 400, o correrlo con GPU.

Uso
---
    python src/validar_profundidad_monocular.py --limite 400
    python src/validar_profundidad_monocular.py --modelo depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf
    python src/validar_profundidad_monocular.py --solo-evaluar   # reusa el CSV cacheado
"""
import argparse
import itertools
from collections import Counter
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from validar_metodo_acmeai import (STICKER_RGB, cargar_json, evaluar,
                                   sticker_size_px)

RADIO_MUESTREO = 9  # pixeles; disco alrededor del keypoint para la mediana de Z

# Las mascaras ___fuse.png de AcmeAI NO son binarias sobre fondo negro: son mapas de
# 3 colores planos (fondo, animal, sticker). Verificado contra los keypoints: el 74-76%
# de los puntos anotados cae dentro de RGB(255,30,249) y el resto justo sobre el borde
# de la silueta. Tomar "todo lo que no es negro" como cuerpo da la imagen entera (99,6%)
# y deja sin efecto tanto el recorte de z_robusto como el area y el volumen.
ANIMAL_RGB = {"b3": (255, 30, 249), "b4": (255, 30, 249)}
ANCHO_DEFECTO = 1024  # ver --ancho


def ruta_cache(modelo):
    """Un cache por modelo: Depth Pro y Depth Anything no son intercambiables y
    no tienen que pisarse el CSV entre corridas."""
    return config.TABLAS / f"acmeai_profundidad_{modelo.split('/')[-1].lower()}.csv"


# --------------------------------------------------------------------------
# profundidad
# --------------------------------------------------------------------------
def cargar_modelo(nombre):
    """Devuelve una funcion imagen_PIL -> (mapa de profundidad en metros, focal en px).

    Se aisla aca para poder cambiar de modelo sin tocar el resto: la unica
    diferencia entre Depth Pro, UniDepth y Depth Anything metric es de que forma
    sale la focal (predicha, o hay que sacarla del EXIF).
    """
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    proc = AutoImageProcessor.from_pretrained(nombre)
    modelo = AutoModelForDepthEstimation.from_pretrained(nombre).eval()
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo.to(dispositivo)
    print(f"  modelo {nombre} en {dispositivo}")

    def inferir(img):
        entradas = proc(images=img, return_tensors="pt").to(dispositivo)
        with torch.no_grad():
            salida = modelo(**entradas)
        post = proc.post_process_depth_estimation(
            salida, target_sizes=[(img.height, img.width)])[0]
        z = post["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
        # Depth Pro predice la focal; los demas no -> se asume un FOV tipico de
        # celular (~65 grados horizontales), que es lo que hay sin EXIF.
        focal = post.get("focal_length", post.get("focallength_px"))
        if focal is None:
            focal = img.width / (2 * np.tan(np.radians(65) / 2))
        else:
            if hasattr(focal, "cpu"):
                focal = focal.cpu()
            focal = float(np.ravel(np.asarray(focal))[0])
        return z, float(focal)

    return inferir


def z_robusto(z, mascara_animal, u, v, radio=RADIO_MUESTREO):
    """Mediana de Z en un disco alrededor de (u,v), restringida al animal.

    Sin la restriccion a la mascara esto falla justo en los puntos que mas
    importan: cruz, pinbone y los de girth estan SOBRE el borde de la silueta, y
    ahi medio disco cae en el fondo, a varios metros de distancia.
    """
    alto, ancho = z.shape
    u, v = int(round(u)), int(round(v))
    if not (0 <= u < ancho and 0 <= v < alto):
        return None
    y0, y1 = max(0, v - radio), min(alto, v + radio + 1)
    x0, x1 = max(0, u - radio), min(ancho, u + radio + 1)
    recorte_z = z[y0:y1, x0:x1]
    recorte_m = mascara_animal[y0:y1, x0:x1]
    valores = recorte_z[recorte_m] if recorte_m.any() else recorte_z
    valores = valores[np.isfinite(valores) & (valores > 0)]
    return float(np.median(valores)) if valores.size else None


def a_3d(u, v, z, focal, cx, cy):
    return np.array([(u - cx) * z / focal, (v - cy) * z / focal, z], float)


# --------------------------------------------------------------------------
# mascaras
# --------------------------------------------------------------------------
def mascara_animal_y_sticker(ruta_mascara, ancho_img, alto_img, sticker_rgb,
                             animal_rgb):
    """Mascara del cuerpo (bool, al tamano de la imagen) + tamano del sticker en px.

    Las mascaras de AcmeAI vienen en RGB y a una resolucion distinta a la de la
    foto. El cuerpo es el color plano del animal (ver ANIMAL_RGB): el sticker se
    pinta aparte y queda naturalmente afuera, y el fondo tambien tiene color
    propio, asi que NO alcanza con descartar el negro.

    Devuelve cuerpo=None si la silueta sale absurda (la foto entera o casi nada),
    que es la senal de que la paleta de ese batch no es la que se supone.
    """
    m = np.array(Image.open(ruta_mascara).convert("RGB"))
    tam_sticker = sticker_size_px(m, sticker_rgb, ancho_img, alto_img)
    cuerpo = (m == animal_rgb).all(axis=-1)
    fraccion = cuerpo.mean()
    if not 0.02 < fraccion < 0.60:
        return None, tam_sticker
    cuerpo = np.array(Image.fromarray(cuerpo.astype(np.uint8) * 255)
                      .resize((ancho_img, alto_img), Image.NEAREST)) > 127
    return cuerpo, tam_sticker


def metricas_de_mascara(cuerpo, z, focal):
    """Area proyectada en m2 y volumen por solido de revolucion en m3.

    El volumen es el proxy clasico: cada columna de pixeles del cuerpo se trata
    como un disco de diametro igual a su altura. Es crudo, pero a diferencia de
    cualquier distancia tiene unidades de masa/densidad -- es la variable que uno
    querria si pudiera medirla.
    """
    ys, xs = np.where(cuerpo)
    if xs.size < 500:
        return None
    zc = np.where(np.isfinite(z) & (z > 0), z, np.nan)
    z_animal = float(np.nanmedian(zc[cuerpo]))
    if not np.isfinite(z_animal):
        return None

    metros_por_px = zc[cuerpo] / focal
    area = float(np.nansum(metros_por_px ** 2))

    mpp = z_animal / focal
    volumen = 0.0
    orden = np.argsort(xs)
    xs_o, ys_o = xs[orden], ys[orden]
    cortes = np.flatnonzero(np.diff(xs_o)) + 1
    for col in np.split(ys_o, cortes):
        h = (col.max() - col.min() + 1) * mpp
        volumen += np.pi / 4 * h * h * mpp
    return {"z_animal_m": z_animal, "area_m2": area, "volumen_m3": volumen}


# --------------------------------------------------------------------------
# extraccion
# --------------------------------------------------------------------------
def procesar_batch(nombre_batch, json_path, mask_dir, patron, extraer, inferir,
                   indice_img, limite=None, ancho_obj=None):
    raiz = config.DATASET_ACMEAI
    d, imgs, nombres = cargar_json(raiz / json_path)
    idx = {n: i for i, n in enumerate(nombres)}
    pares = list(itertools.combinations(sorted(idx), 2))
    mask_dir = raiz / mask_dir

    filas, vistas, t_ini = [], 0, time.time()
    descartes = Counter()
    for ann in d["annotations"]:
        if limite and vistas >= limite:
            break
        meta = imgs[ann["image_id"]]
        fn = meta["file_name"]
        m = patron.match(fn)
        if not m or ann["num_keypoints"] < len(nombres):
            descartes["patron o keypoints incompletos"] += 1
            continue
        ruta_img, ruta_mask = indice_img.get(fn), mask_dir / f"{fn}___fuse.png"
        if ruta_img is None or not ruta_mask.exists():
            descartes["falta la imagen o la mascara"] += 1
            continue

        # Los keypoints del json estan en el sistema de coordenadas que declara el
        # PROPIO json, que no siempre es el tamano del jpg que hay en disco: en B3 el
        # json dice 4160x3120 y el archivo es de 1900x1425. Escalar por el tamano del
        # archivo tira los puntos afuera de la imagen y descarta el batch entero en
        # silencio, asi que la escala se calcula contra meta["width"].
        #
        # Redimensionar antes de inferir es exacto, no una aproximacion: escalar u, v
        # y f por el mismo factor deja X/Z e Y/Z intactos y Z no cambia, asi que las
        # distancias 3D en metros son las mismas. Y casi todo el costo por imagen
        # (upsample del mapa de profundidad, mascara, area, volumen) escala con el
        # tamano de la imagen, no con el del modelo.
        ancho_dest = ancho_obj or meta["width"]
        esc = ancho_dest / meta["width"]
        alto_dest = round(meta["height"] * esc)
        img = Image.open(ruta_img).convert("RGB")
        if img.size != (ancho_dest, alto_dest):
            img = img.resize((ancho_dest, alto_dest), Image.BILINEAR)

        cuerpo, tam_sticker = mascara_animal_y_sticker(
            ruta_mask, img.width, img.height, STICKER_RGB[nombre_batch],
            ANIMAL_RGB[nombre_batch])
        if tam_sticker is None:
            descartes["no se encontro el sticker"] += 1
            continue
        if cuerpo is None:
            descartes["silueta de tamano absurdo (paleta inesperada)"] += 1
            continue

        z, focal = inferir(img)
        met = metricas_de_mascara(cuerpo, z, focal)
        if met is None:
            descartes["mascara demasiado chica o Z invalida"] += 1
            continue

        cx, cy = img.width / 2, img.height / 2
        kp = np.array(ann["keypoints"], float).reshape(len(nombres), 3)[:, :2] * esc
        puntos = {}
        for n, i in idx.items():
            zi = z_robusto(z, cuerpo, kp[i, 0], kp[i, 1])
            if zi is None:
                break
            puntos[n] = a_3d(kp[i, 0], kp[i, 1], zi, focal, cx, cy)
        if len(puntos) < len(idx):
            descartes["keypoints fuera de la imagen"] += 1
            continue

        aid, peso, sexo = extraer(m.groups())
        fila = {"batch": nombre_batch, "grupo": f"{nombre_batch}_{aid}_{peso}_{sexo}",
                "peso_kg": float(peso), "focal_px": focal,
                "sticker_px": tam_sticker, **met}
        for a, b in pares:
            # 3D metrico (sin marcador) y 2D calibrado por sticker (linea base)
            fila[f"m_{a}_{b}"] = float(np.linalg.norm(puntos[a] - puntos[b]))
            fila[f"s_{a}_{b}"] = float(np.linalg.norm(kp[idx[a]] - kp[idx[b]])) / tam_sticker
        filas.append(fila)
        vistas += 1
        if vistas % 25 == 0:
            seg = time.time() - t_ini
            print(f"    {nombre_batch}: {vistas} imagenes  ({seg / vistas:.2f} s/img,"
                  f" {seg / 60:.1f} min acumulados)", flush=True)

    # Los descartes se imprimen siempre: un batch que se cae entero por un
    # desajuste de coordenadas tiene que ser visible, no silencioso.
    for motivo, n in descartes.most_common():
        print(f"    descartadas por {motivo}: {n}")
    print(f"  {nombre_batch}: {len(filas)} filas utiles "
          f"({(time.time() - t_ini) / 60:.1f} min)", flush=True)
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# diagnostico: la escala de la red contra la escala del sticker
# --------------------------------------------------------------------------
def diagnostico_escala(df):
    """La pregunta del proyecto, en una sola correlacion.

    El sticker tiene tamano real fijo y desconocido S. Su tamano en pixeles es
    entonces  sticker_px = f * S / Z  ->  la escala "px por metro" a la
    profundidad del animal es proporcional a sticker_px.
    La red de profundidad estima esa misma cantidad como  f / Z_animal.
    Si las dos correlacionan fuerte, la red mide lo que mide el sticker y el
    sticker se puede tirar. Si no, el sticker hace falta y hay que decirlo.
    """
    escala_red = (df["focal_px"] / df["z_animal_m"]).to_numpy(float)
    escala_sticker = df["sticker_px"].to_numpy(float)
    ok = np.isfinite(escala_red) & np.isfinite(escala_sticker) & (escala_sticker > 0)
    escala_red, escala_sticker = escala_red[ok], escala_sticker[ok]
    r = np.corrcoef(escala_red, escala_sticker)[0, 1]
    r_log = np.corrcoef(np.log(escala_red), np.log(escala_sticker))[0, 1]
    razon = escala_red / escala_sticker
    print("\n=== DIAGNOSTICO: escala de la red vs escala del sticker ===")
    print("  (el sticker se usa SOLO como regla de validacion, no entra al modelo)")
    print(f"  r  = {r:+.3f}   r(log-log) = {r_log:+.3f}   n = {ok.sum()}")
    print(f"  dispersion del factor red/sticker: CV = "
          f"{razon.std() / razon.mean() * 100:.1f}%")
    print("  Lectura: r > 0,90 y CV < 10% -> el sticker es prescindible.")
    print("           r < 0,70            -> la red no recupera la escala aca.")


def main(args):
    if not config.DATASET_ACMEAI.exists():
        print(f"No se encontro el dataset en {config.DATASET_ACMEAI}")
        return

    cache = ruta_cache(args.modelo)
    if args.solo_evaluar:
        if not cache.exists():
            print(f"No hay cache en {cache}; corre sin --solo-evaluar primero.")
            return
        df = pd.read_csv(cache)
    else:
        inferir = cargar_modelo(args.modelo)
        # Las fotos no estan siempre al lado del json: se indexa una sola vez.
        indice_img = {p.name: p for p in config.DATASET_ACMEAI.rglob("*.jpg")}
        print(f"  {len(indice_img)} imagenes indexadas")
        pat_b3 = re.compile(r"^(\d+)_s_(\d+)_([MF])\.jpg$", re.IGNORECASE)
        pat_b4 = re.compile(r"^(\d+)_(b4-\d+)_s_(\d+)_([MF])\.jpg$", re.IGNORECASE)
        print("Extrayendo profundidad (esto es lo lento)...")
        df3 = procesar_batch("b3", "Vector/B3/Side/data/COCO_Side.json",
                             "Pixel/B3/annotations", pat_b3,
                             lambda g: (g[0], g[1], g[2]), inferir,
                             indice_img, args.limite, args.ancho)
        df4 = procesar_batch("b4", "Vector/B4/Side/data/coco_b4_side.json",
                             "Pixel/B4/Side/annotations", pat_b4,
                             lambda g: (g[0], g[2], g[3]), inferir,
                             indice_img, args.limite, args.ancho)
        df = pd.concat([df3, df4], ignore_index=True)
        config.TABLAS.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        print(f"\n-> cacheado en {cache}")

    if df.empty:
        print("Ninguna imagen sobrevivio la extraccion.")
        return

    print(f"Modelo de profundidad: {args.modelo}  (ancho {args.ancho} px)")
    diagnostico_escala(df)

    m = sorted(c for c in df.columns if c.startswith("m_"))
    s = sorted(c for c in df.columns if c.startswith("s_"))
    vol = ["area_m2", "volumen_m3"]
    franjas = [(0, 100), (100, 200), (200, 300), (300, 10_000)]

    evaluar(df, s, "LINEA BASE -- 36 distancias calibradas por sticker", franjas)
    evaluar(df, m, "SIN MARCADOR -- 36 distancias 3D en metros reales", franjas)
    evaluar(df, m + vol, "SIN MARCADOR -- distancias 3D + area y volumen metricos", franjas)
    evaluar(df, vol, "SIN MARCADOR -- solo area y volumen metricos", franjas)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="apple/DepthPro-hf")
    ap.add_argument("--limite", type=int, help="imagenes por batch (prueba rapida)")
    ap.add_argument("--ancho", type=int, default=ANCHO_DEFECTO,
                    help="redimensionar el ancho a N px antes de inferir "
                         "(exacto geometricamente; 0 = dejar el original)")
    ap.add_argument("--solo-evaluar", action="store_true")
    main(ap.parse_args())
