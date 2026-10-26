"""Tres sets de fotos para DEMOSTRAR el modelo (no para medirlo). Salida: `demo/<set>/` con las fotos y un `pesos.csv`.

  demo_6-10_mejores_v2   12 fotos de la jornada del 6/10 que mejor dio el modelo v2 (una por animal). **Elegidas por su
                         resultado**: sirven para mostrar cómo se ve cuando acierta, no para estimar qué tan bien anda.
  demo_bangladesh_v1     20 fotos del dataset de Bangladesh (animales del split de TEST, que el v1 no vio), una por animal y
                         repartidas por peso (no elegidas por resultado). Para ver el modelo v1 en su terreno.
  demo_propio_v2         15 fotos del 8/10 y 22/10, una por animal, laterales completas y repartidas por peso. **El v2 se
                         entrenó con estas vacas**: muestra el modelo en su terreno, no es una prueba.

Uso:  uv run --extra cpu python 53_armar_demos.py
Después:  cd ../estimador && .venv\\Scripts\\python probar.py --carpeta ../demo/<set> --pesos ../demo/<set>/pesos.csv --web?  (ver README de demo/)
"""

import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image

import config

sys.path.insert(0, str(config.ROOT.parent / "estimador"))
from estimador import EstimadorPeso  # noqa: E402

DEMO = config.ROOT.parent / "demo"
MODELOS = config.ROOT.parent / "estimador"


def repartidos(df, n, col="peso_kg"):
    """n filas igualmente espaciadas por `col` (desde la más liviana a la más pesada)."""
    d = df.sort_values(col).reset_index(drop=True)
    return d.iloc[np.linspace(0, len(d) - 1, n).round().astype(int)]


def escribir(nombre, filas):
    carpeta = DEMO / nombre
    if carpeta.exists():
        shutil.rmtree(carpeta)
    carpeta.mkdir(parents=True)
    for f in filas:
        shutil.copy2(f["origen"], carpeta / f["foto"])
    pd.DataFrame(filas)[["foto", "peso_kg", "animal"]].to_csv(carpeta / "pesos.csv", index=False)
    pesos = [f["peso_kg"] for f in filas]
    print(f"{nombre}: {len(filas)} fotos, {min(pesos):.0f}-{max(pesos):.0f} kg")


def set_6_10_mejores():
    fotos = pd.read_csv(config.PRUEBA / "fotos.csv", dtype={"animal_id": str}, keep_default_na=False)
    fotos = fotos[fotos.uso == "incluida"].copy()
    est = EstimadorPeso(MODELOS / "modelos_v2")
    from estimador import LADO_MAX  # noqa: F401  (el módulo reduce solo)
    errores = []
    for f in fotos.itertuples():
        r = est.estimar(Image.open(config.PRUEBA / f.archivo_curado).convert("RGB"))
        errores.append(abs(r["peso_kg"] - f.peso_kg) / f.peso_kg * 100 if r["peso_kg"] else np.inf)
    fotos["error_v2"] = errores
    mejores = fotos.sort_values("error_v2").drop_duplicates("animal_id").head(12)
    print("  errores del v2 en estas 12:", mejores.error_v2.round(1).tolist())
    return [{"foto": f"{f.archivo_curado.split('/')[-1]}", "peso_kg": f.peso_kg, "animal": f.animal_id,
             "origen": config.PRUEBA / f.archivo_curado} for f in mejores.itertuples()]


def set_bangladesh():
    man = pd.read_parquet(config.TABLAS / "manifest.parquet")
    t = man[man.split == "test"].sort_values("peso_kg").reset_index(drop=True)
    # 20 pesos objetivo repartidos entre el percentil 2 y el 98; para cada uno, la foto más cercana de un animal NO elegido
    objetivos = np.quantile(t.peso_kg, np.linspace(0.02, 0.98, 20))
    usados, elegidas = set(), []
    for objetivo in objetivos:
        for i in (t.peso_kg - objetivo).abs().sort_values().index:
            if t.at[i, "animal_id"] not in usados:
                usados.add(t.at[i, "animal_id"])
                elegidas.append(i)
                break
    filas = []
    for f in t.loc[elegidas].itertuples():
        nombre = f.image_id.replace("/", "__")
        filas.append({"foto": nombre, "peso_kg": f.peso_kg, "animal": f.animal_id,
                      "origen": config.DATA / "pose" / "images_sin_sticker" / "test" / nombre})
    return filas


def set_propio():
    fotos = pd.read_csv(config.PROPIO / "fotos.csv", dtype={"animal_id": str})
    fotos = fotos[(fotos.uso == "incluida") & fotos.lateral_completo].sort_values("score", ascending=False)
    uno = fotos.drop_duplicates("animal_id")
    sel = repartidos(uno, 15)
    return [{"foto": f.archivo_curado.split("/")[-1], "peso_kg": f.peso_kg, "animal": f.animal_id,
             "origen": config.PROPIO / f.archivo_curado} for f in sel.itertuples()]


def main():
    escribir("demo_6-10_mejores_v2", set_6_10_mejores())
    escribir("demo_bangladesh_v1", set_bangladesh())
    escribir("demo_propio_v2", set_propio())


if __name__ == "__main__":
    main()
