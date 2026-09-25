"""Fase 3/4c: arma un dataset unificado AcmeAI + CID + Mendeley para cubrir todo el rango de peso.

Motivo (medido): la CNN entrenada solo con AcmeAI **satura en 267 kg** porque el entrenamiento tiene 1.394
imágenes entre 150 y 200 kg y solo 22 por encima de 300. Por eso subestima un 31% a los animales grandes y falla
por completo en Mendeley (341-644 kg). Sumar CID (150-816 kg) y Mendeley (341-644 kg) multiplica por 4 los datos
del rango alto, que es justo el rango que le interesa al productor argentino (novillos de 400-500 kg).

Qué hace:
  - reúne las tres fuentes con un formato común (una foto lateral por animal en CID y Mendeley)
  - asigna split POR ANIMAL en cada fuente, respetando el split ya fijado de AcmeAI para poder comparar
  - deja las imágenes de CID/Mendeley reducidas a 1024 px (AcmeAI ya está preparado, no se duplica)
  - guarda los keypoints en coordenadas de la imagen ORIGINAL, igual que el manifest de AcmeAI, así el
    entrenamiento no necesita ningún caso especial

Salidas: data/tablas/manifest_unificado.parquet, data/tablas/keypoints_unificado.parquet,
         data/pose/extra/<split>/  (solo CID y Mendeley)
Uso:  uv run python 29_dataset_unificado.py
"""
import importlib

import numpy as np
import pandas as pd
from PIL import Image

import config
from common import splits
from common.acmeai import KP_CANONICOS

LADO = 1024
c27 = importlib.import_module("27_cross_cid")


def preparar_imagen(ruta_origen, destino):
    """Reduce a 1024 px de lado mayor (mismo criterio que 06a_preparar_pose.py). Devuelve (ancho, alto) ORIGINALES."""
    im = Image.open(ruta_origen).convert("RGB")
    ancho, alto = im.size
    if not destino.exists():
        f = LADO / max(im.size)
        if f < 1:
            im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
        destino.parent.mkdir(parents=True, exist_ok=True)
        im.save(destino, quality=93)
    return ancho, alto


def main():
    # --- AcmeAI: ya preparado, se referencia tal cual ---
    acme = pd.read_parquet(config.TABLAS / "manifest.parquet")
    kp_acme = pd.read_parquet(config.TABLAS / "keypoints_pred.parquet").set_index("image_id")
    acme = acme[acme.image_id.isin(kp_acme.index)].copy()
    acme["origen"], acme["dir_imagenes"] = "acmeai", "images_sin_sticker"

    partes, kps = [acme[["image_id", "origen", "dir_imagenes", "animal_id", "peso_kg", "split", "ancho", "alto"]]], \
                  [kp_acme[[f"{n}_{c}" for n in KP_CANONICOS for c in ("x", "y")]]]

    # --- CID: una lateral por animal (la de mayor relación ancho/alto de los keypoints) ---
    man_cid = c27.manifest_cid()
    kp_cid = c27.keypoints_cid(man_cid)
    lat = c27.elegir_lateral(man_cid, kp_cid)
    filas = []
    for r in lat.itertuples():
        nombre = r.Index.replace("/", "__")
        ancho, alto = preparar_imagen(r.ruta_imagen, config.DATA / "pose" / "extra" / "tmp" / nombre)
        filas.append({"image_id": r.Index, "origen": "cid", "animal_id": r.animal_id,
                      "peso_kg": r.peso_kg, "ancho": ancho, "alto": alto, "nombre": nombre})
    cid = pd.DataFrame(filas)
    cid["split"] = cid.animal_id.map(splits.asignar_split(cid.animal_id))
    kps.append(lat[[f"{n}_{c}" for n in KP_CANONICOS for c in ("x", "y")]].rename_axis("image_id"))

    # --- Mendeley ---
    m26 = importlib.import_module("26_adaptacion")
    man_mend = pd.read_parquet(config.TABLAS / "manifest_mendeley.parquet")
    kp_mend = m26.keypoints_mendeley()
    man_mend = man_mend[man_mend.image_id.isin(kp_mend.index)]
    filas = []
    for r in man_mend.itertuples():
        nombre = r.image_id.replace("/", "__").replace(".png", ".jpg")
        ancho, alto = preparar_imagen(r.ruta_imagen, config.DATA / "pose" / "extra" / "tmp" / nombre)
        filas.append({"image_id": r.image_id, "origen": "mendeley", "animal_id": r.animal_id,
                      "peso_kg": r.peso_kg, "ancho": ancho, "alto": alto, "nombre": nombre})
    mend = pd.DataFrame(filas)
    mend["split"] = mend.animal_id.map(splits.asignar_split(mend.animal_id))
    kps.append(kp_mend.loc[man_mend.image_id, [f"{n}_{c}" for n in KP_CANONICOS for c in ("x", "y")]])

    # mover cada imagen nueva a la carpeta de su split (el entrenamiento busca en <dir>/<split>/<nombre>)
    for df in (cid, mend):
        for r in df.itertuples():
            origen = config.DATA / "pose" / "extra" / "tmp" / r.nombre
            destino = config.DATA / "pose" / "extra" / r.split / r.nombre
            destino.parent.mkdir(parents=True, exist_ok=True)
            if origen.exists():
                origen.replace(destino)
        df["dir_imagenes"] = "extra"
        # el entrenamiento arma el nombre como image_id.replace('/','__'); en Mendeley el archivo es .jpg
        df["image_id"] = df.nombre.str.replace("__", "/", regex=False)
    partes += [cid[["image_id", "origen", "dir_imagenes", "animal_id", "peso_kg", "split", "ancho", "alto"]],
               mend[["image_id", "origen", "dir_imagenes", "animal_id", "peso_kg", "split", "ancho", "alto"]]]

    man = pd.concat(partes, ignore_index=True)
    kp = pd.concat(kps)
    kp.index = man.image_id.values  # mismo orden de concatenación
    assert man.image_id.is_unique, "image_id repetido entre fuentes"
    man.to_parquet(config.TABLAS / "manifest_unificado.parquet", index=False)
    kp.rename_axis("image_id").reset_index().to_parquet(config.TABLAS / "keypoints_unificado.parquet", index=False)

    print(f"{len(man)} imágenes de {man.animal_id.nunique()} animales")
    print(man.groupby("origen").agg(n=("image_id", "size"), peso_min=("peso_kg", "min"),
                                    peso_max=("peso_kg", "max")).to_string())
    man["banda"] = pd.cut(man.peso_kg, [0, 100, 150, 200, 250, 300, 10_000], right=False)
    print("\nimágenes de ENTRENAMIENTO por banda (antes / ahora):")
    tr = man[man.split == "train"]
    comp = pd.DataFrame({"solo_acmeai": tr[tr.origen == "acmeai"].groupby("banda", observed=True).size(),
                         "unificado": tr.groupby("banda", observed=True).size()}).fillna(0).astype(int)
    print(comp.to_string())
    print(f"\n-> {config.TABLAS / 'manifest_unificado.parquet'}")


if __name__ == "__main__":
    main()
