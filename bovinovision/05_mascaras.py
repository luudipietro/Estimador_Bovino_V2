"""Etapa 4 (parte GT): extrae de las máscaras de AcmeAI el área/bbox de la vaca y la geometría del sticker.
-> data/tablas/mascaras.parquet   (una fila por imagen; clave image_id)
Uso:  uv run python 05_mascaras.py
"""
import numpy as np
import pandas as pd
from PIL import Image

import config
from common import masks


def main():
    m = pd.read_parquet(config.TABLAS / "manifest.parquet")
    filas = []
    for i, r in enumerate(m.itertuples(), 1):
        rgba = np.array(Image.open(config.ACMEAI / r.ruta_mascara).convert("RGBA"))
        vaca, sticker = masks.separar(rgba)
        filas.append({"image_id": r.image_id, **masks.estadisticas(vaca, sticker)})
        if i % 1000 == 0:
            print(f"  {i}/{len(m)}")
    df = pd.DataFrame(filas)
    df.to_parquet(config.TABLAS / "mascaras.parquet", index=False)

    d = df.merge(m[["image_id", "lote", "ancho"]], on="image_id")
    print(f"{len(df)} máscaras. Sin sticker: {d['sticker_diam_px'].isna().sum()} | "
          f"con >1 componente de sticker: {(d['n_comp_sticker'] > 1).sum()} | "
          f"redondez<0.9 (ocluido/recortado): {(d['sticker_redondez'] < 0.9).sum()}")
    d["sticker_rel"] = d["sticker_diam_px"] / d["ancho"]  # diámetro relativo al ancho de la imagen
    print(d.groupby("lote")[["sticker_diam_px", "sticker_rel", "area_vaca_px"]].describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
