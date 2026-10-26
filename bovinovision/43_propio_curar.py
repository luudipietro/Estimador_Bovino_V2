"""Dataset propio, etapa 4: asigna el split por animal y arma `dataset_propio/curado/` con las fotos que se usan.

  - split por ANIMAL (nunca por foto) con el mismo criterio del resto del proyecto (common.splits, seed fija)
  - copia cada foto `incluida` a  curado/<split>/<animal_id>__<pesaje_id>__<n>.jpg  reducida a 1024 px de lado mayor
    (mismo formato que 29_dataset_unificado.py; también convierte los HEIC del iPhone 14). El peso NO va en el nombre:
    sale de fotos.csv / pesajes.csv. Los originales no se tocan.
  - imprime el balance por banda de peso contra lo que pide RECOLECCION_DATASET.md

Uso:  uv run python 43_propio_curar.py
"""
import pandas as pd
import pillow_heif
from PIL import Image, ImageOps

import config
from common import splits

pillow_heif.register_heif_opener()

LADO = 1024
BANDAS_META = [(0, 150, 50), (150, 250, 60), (250, 350, 70), (350, 450, 70), (450, 550, 50), (550, 10_000, None)]


def reducir(origen, destino):
    im = ImageOps.exif_transpose(Image.open(origen)).convert("RGB")
    f = LADO / max(im.size)
    if f < 1:
        im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
    destino.parent.mkdir(parents=True, exist_ok=True)
    im.save(destino, quality=93)


def main():
    df = pd.read_csv(config.PROPIO / "fotos.csv", dtype={"animal_id": str})
    inc = df.uso == "incluida"

    sp = splits.asignar_split(df[inc].animal_id)
    df["split"] = df.animal_id.map(sp).where(inc)
    splits.verificar_sin_fuga(df[inc])

    df["archivo_curado"] = None
    for animal, g in df[inc].groupby("animal_id"):
        for n, i in enumerate(g.sort_values("hora").index, 1):
            r = df.loc[i]
            rel = f"curado/{r.split}/{animal}__{r.pesaje_id}__{n}.jpg"
            destino = config.PROPIO / rel
            if not destino.exists():
                reducir(config.PROPIO / r.foto, destino)
            df.at[i, "archivo_curado"] = rel
    df.to_csv(config.PROPIO / "fotos.csv", index=False)

    # pesajes.csv: cuántas fotos se usan de cada pesaje y a qué split cae el animal
    pes = pd.read_csv(config.PROPIO / "pesajes.csv", dtype={"ide": str, "animal_id": str})
    pes = pes.drop(columns=[c for c in ("n_fotos_incluidas", "split") if c in pes.columns])
    pes["n_fotos_incluidas"] = pes.pesaje_id.map(df[inc].groupby("pesaje_id").size()).fillna(0).astype(int)
    pes["split"] = pes.animal_id.map(sp)
    pes.to_csv(config.PROPIO / "pesajes.csv", index=False)

    # --- sin archivos huérfanos: todo lo que hay en curado/ debe estar en fotos.csv ---
    en_disco = {p.relative_to(config.PROPIO).as_posix() for p in (config.PROPIO / "curado").rglob("*.jpg")}
    huerfanos = en_disco - set(df.archivo_curado.dropna())
    for h in huerfanos:
        (config.PROPIO / h).unlink()
    print(f"curado/: {len(en_disco) - len(huerfanos)} fotos ({len(huerfanos)} huérfanas borradas)")

    animales = df[inc].drop_duplicates("animal_id")
    print("\nanimales y fotos por split:")
    print(pd.concat([animales.groupby("split").size().rename("animales"), df[inc].groupby("split").size().rename("fotos")],
                    axis=1).to_string())

    print("\nbalance por banda de peso (animales con foto usable) vs meta de RECOLECCION_DATASET.md (para 300 animales):")
    for lo, hi, meta in BANDAS_META:
        n = int(((animales.peso_kg >= lo) & (animales.peso_kg < hi)).sum())
        print(f"  {lo:>4}-{hi if hi < 10_000 else '+':<5} kg: {n:>3} animales" + (f"   (meta {meta})" if meta else ""))
    print(f"  rango real: {animales.peso_kg.min():.0f}-{animales.peso_kg.max():.0f} kg | mediana {animales.peso_kg.median():.0f}")
    print(f"  baseline 'predice la mediana' sobre estos animales: MAPE "
          f"{((animales.peso_kg - animales.peso_kg.median()).abs() / animales.peso_kg).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
