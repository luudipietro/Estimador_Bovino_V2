"""Etapa 6: keypoints -> features geométricas (distancias px, ratios sin escala, distancias/sticker).
-> data/tablas/features_gt.parquet (a partir de keypoints anotados a mano). Más adelante se repite con keypoints predichos.
Uso:  uv run python 08_features_2d.py [--kp keypoints_gt]
"""
import argparse

import pandas as pd

import config
from common import features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", default="keypoints_gt", help="tabla de keypoints en data/tablas (sin extensión)")
    args = ap.parse_args()

    kp = pd.read_parquet(config.TABLAS / f"{args.kp}.parquet").set_index("image_id")
    msk = pd.read_parquet(config.TABLAS / "mascaras.parquet").set_index("image_id")
    man = pd.read_parquet(config.TABLAS / "manifest.parquet").set_index("image_id")

    d = features.distancias_px(kp)
    r = features.ratios(d)
    s = features.normalizadas_por_sticker(d, msk.loc[kp.index, "sticker_diam_px"])
    out = pd.concat([man.loc[kp.index, ["animal_id", "lote", "split", "peso_kg"]], d, r, s], axis=1)
    destino = config.TABLAS / args.kp.replace("keypoints", "features")
    out.reset_index().to_parquet(f"{destino}.parquet", index=False)
    print(f"{len(out)} filas -> {destino}.parquet | features: {d.shape[1]} d, {r.shape[1]} r, {s.shape[1]} s")
    print(f"NaN por familia: d={int(d.isna().sum().sum())} r={int(r.isna().sum().sum())} s={int(s.isna().sum().sum())}")


if __name__ == "__main__":
    main()
