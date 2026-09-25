"""Fase 3/4: incorpora el dataset Mendeley (72 bovinos adultos de Mongolia Interior, 341-644 kg) como
manifest propio, para cubrir el rango de peso alto que AcmeAI casi no tiene (solo 35 imágenes sobre 300 kg).

Es el dataset que permite responder si el modelo sirve fuera de Bangladesh: otra raza, otro país, otra cámara y,
sobre todo, otro rango de peso. Trae además las medidas reales en cm (altura a la cruz, largo, perímetro
torácico), que sirven como referencia de cuánto vale una medida física.

Salida: data/tablas/manifest_mendeley.parquet (mismas columnas clave que el manifest de AcmeAI: image_id,
animal_id, peso_kg, split, ruta_imagen) + las medidas en cm.
Uso:  uv run python 23_mendeley_manifest.py
"""
import pandas as pd

import config
from common import splits

BASE = config.ROOT.parent / "modelo-peso" / "data" / "crudo" / "Cattle side view and back view dataset" / "Cattle side and back view images"
COLS_CM = {"Oblique body length (cm)": "largo_cm", "Withers height(cm)": "altura_cruz_cm",
           "Heart girth(cm)": "perimetro_toracico_cm", "Hip length (cm)": "largo_grupa_cm",
           "Body weight (kg)": "peso_kg"}


def main():
    x = pd.read_excel(BASE / "measurements.xlsx")
    x.columns = [c.strip() for c in x.columns]
    ren = {k.strip(): v for k, v in COLS_CM.items()}
    faltan = [k for k in ren if k not in x.columns]
    assert not faltan, f"columnas no encontradas en measurements.xlsx: {faltan} (hay: {x.columns.tolist()})"
    x = x.rename(columns=ren)

    lateral = BASE / "side view"
    filas = []
    for _, r in x.iterrows():
        num = int(r["Num"])
        img = lateral / f"{num}.png"
        if not img.exists():
            print(f"  aviso: falta {img.name}")
            continue
        filas.append({"image_id": f"MEND/{num}.png", "animal_id": f"MEND{num}", "lote": "MENDELEY",
                      "peso_kg": float(r["peso_kg"]), "sexo": "?",
                      "ruta_imagen": str(img.resolve()),
                      **{v: float(r[v]) for v in ("largo_cm", "altura_cruz_cm", "perimetro_toracico_cm", "largo_grupa_cm")}})
    df = pd.DataFrame(filas)
    df["split"] = df["animal_id"].map(splits.asignar_split(df["animal_id"]))
    splits.verificar_sin_fuga(df)
    df.to_parquet(config.TABLAS / "manifest_mendeley.parquet", index=False)

    print(f"{len(df)} animales -> {config.TABLAS / 'manifest_mendeley.parquet'}")
    print(df.groupby("split").agg(n=("image_id", "size"), peso_min=("peso_kg", "min"),
                                  peso_med=("peso_kg", "median"), peso_max=("peso_kg", "max")).to_string())
    acme = pd.read_parquet(config.TABLAS / "manifest.parquet")
    print(f"\nsolapamiento de rangos: AcmeAI {acme.peso_kg.min():.0f}-{acme.peso_kg.max():.0f} kg "
          f"({(acme.peso_kg > 300).sum()} imágenes sobre 300 kg) | Mendeley {df.peso_kg.min():.0f}-{df.peso_kg.max():.0f} kg")
    print(f"animales de AcmeAI dentro del rango de Mendeley: {(acme.peso_kg >= df.peso_kg.min()).sum()} imágenes")


if __name__ == "__main__":
    main()
