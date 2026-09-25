"""Etapa 8 (puerta): ¿la profundidad métrica recupera la escala real con la precisión necesaria?

Idea: el sticker físico tiene el MISMO tamaño real en las 300 imágenes (no lo conocemos en cm, pero es constante).
Con el pinhole (diam_real = diam_px * z / focal), si la profundidad y la focal fueran perfectas, el "diámetro real
estimado" del sticker sería igual en todas las imágenes. La dispersión de ese valor entre imágenes ES el error de
escala que después distorsiona el peso — es la cantidad que se simuló en 10a_sensibilidad_escala.py.

Umbral (de 10a): seguir sin marcador solo si error de escala mediano <= ~3.5% y P90 <= ~8% (equivalente a σ≈5%).

Segundo diagnóstico (más débil, no decide nada): consistencia de z entre keypoints de la misma imagen. En vista
lateral el animal es casi plano, así que z en los 9 keypoints debería ser parecida a z en el sticker/la vaca;
mucha dispersión interna sugiere que el mapa de profundidad no es confiable punto a punto (más allá del error
de escala global que mide la prueba principal).

Uso:  uv run python 11_depth_scale_check.py --modelo pro
"""
import argparse

import numpy as np
import pandas as pd

import config
from common.acmeai import KP_CANONICOS

UMBRAL_MEDIANA = 3.5
UMBRAL_P90 = 8.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="pro")
    a = ap.parse_args()

    d = pd.read_parquet(config.TABLAS / f"depth_{a.modelo}.parquet").set_index("image_id")
    m = pd.read_parquet(config.TABLAS / "mascaras.parquet").set_index("image_id")
    df = d.join(m[["sticker_diam_px", "sticker_redondez", "n_comp_sticker"]], how="inner")
    antes = len(df)
    df = df[(df.sticker_redondez >= 0.9) & (df.n_comp_sticker == 1) & df.z_sticker_m.notna()]
    print(f"{len(df)}/{antes} imágenes con sticker limpio (redondo, sin ocluir) y profundidad válida")

    if df.focal_px_est.notna().any():
        focal = df.focal_px_est  # Depth Pro: focal propia por imagen
    else:
        focal = pd.Series(1600.0, index=df.index)  # placeholder si el modelo no estima focal (ver aviso abajo)
        print("AVISO: este modelo no estima focal; usando un valor fijo de referencia (revisar con EXIF real)")

    diam_real_m = df.sticker_diam_px * df.z_sticker_m / focal
    mediana = diam_real_m.median()
    error_pct = (diam_real_m / mediana - 1).abs() * 100

    print(f"\nDiámetro real estimado del sticker (constante en la realidad): "
          f"mediana {mediana * 100:.2f} cm, CV {diam_real_m.std() / mediana * 100:.1f}%")
    print(f"Error de escala |estimado/mediana - 1|: mediana {error_pct.median():.2f}% | "
          f"P90 {error_pct.quantile(0.9):.2f}% | máx {error_pct.max():.2f}%")

    ok = error_pct.median() <= UMBRAL_MEDIANA and error_pct.quantile(0.9) <= UMBRAL_P90
    print(f"\nUmbral: mediana <= {UMBRAL_MEDIANA}% y P90 <= {UMBRAL_P90}%  ->  "
          f"{'PASA: la profundidad podría reemplazar al sticker' if ok else 'NO PASA: el sticker sigue siendo necesario'}")

    # Diagnóstico secundario: consistencia de z entre los 9 keypoints de una misma imagen (vista lateral ~ plana)
    zk = df[[f"z_{k}_m" for k in KP_CANONICOS]]
    cv_interno = (zk.std(axis=1) / zk.mean(axis=1) * 100)
    print(f"\nDispersión de z entre keypoints de la misma imagen (debería ser baja, vista lateral ~ plana): "
          f"CV mediano {cv_interno.median():.1f}% (P90 {cv_interno.quantile(0.9):.1f}%)")

    df.assign(diam_real_cm=diam_real_m * 100, error_escala_pct=error_pct).reset_index().to_csv(
        config.RUNS / f"depth_scale_check_{a.modelo}.csv", index=False)


if __name__ == "__main__":
    main()
