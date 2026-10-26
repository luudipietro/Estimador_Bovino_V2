"""Dataset propio, etapa 3b: decide qué fotos de vaca se usan (`incluida`) y cuáles se descartan, y por qué.

Parte de las señales medidas en 42a_propio_features.py (rápido: se puede reajustar los umbrales sin volver a correr
los modelos) y de fotos.csv del pareo. Una foto es USABLE si cumple todo lo que pide RECOLECCION_DATASET.md:
un solo animal dominante, cuerpo completo dentro del cuadro, vista lateral, tamaño razonable, keypoints confiables y
peso de balanza confiable. De las usables se dejan hasta FOTOS_POR_ANIMAL, las de mejor score y separadas en el
tiempo (las ráfagas de 0,4 s son casi idénticas y no aportan información nueva). Nada se borra: todo queda
registrado en fotos.csv con su motivo.

Reescribe dataset_propio/fotos.csv a partir de las columnas del pareo (idempotente).
Uso:  uv run python 42_propio_calidad.py
"""
import numpy as np
import pandas as pd

import config

FOTOS_POR_ANIMAL = 3
SEPARACION_MIN_S = 0.35         # entre fotos elegidas del mismo animal: el iPhone dispara cada ~0,4 s, se admiten fotos de una misma ráfaga
DOMINANCIA_MIN = 2.0            # el animal principal debe tener >= 2x el area del segundo
ASPECTO_LATERAL_MIN = 1.05      # ancho/alto de los keypoints: debajo de esto es frontal/trasera/asomando por la manga
                                # (calibrado mirando fotos: >=1,1 son laterales completas; <0,9 son todas malas)
ANCHO_CAJA_MIN = 0.30           # el animal debe ocupar al menos 30% del ancho del cuadro
CONF_DET_MIN = 0.50
CONF_KP_MIN = 0.50

PAREO = ["foto", "dia", "hora", "ms", "pesaje_id", "animal_id", "peso_kg", "peso_confiable", "dt_s", "telefono",
         "ancho", "alto", "focal_35mm"]


def motivo_geometrico(f):
    """Primer motivo por el que una foto NO sirve (None si sirve). El orden es el de gravedad."""
    if pd.isna(f.conf_det) or f.n_vacas == 0:
        return "sin_deteccion"
    if f.dominancia < DOMINANCIA_MIN:
        return "varios_animales"
    # tocar el borde por la cola no es grave si el pose ve los 9 puntos; sí lo es si hay partes que no se ven
    if f.caja_toca_borde and f.conf_kp_min < CONF_KP_MIN:
        return "cortada_borde"
    if f.caja_ancho_frac < ANCHO_CAJA_MIN:
        return "muy_chica"
    if f.aspecto_kp < ASPECTO_LATERAL_MIN:
        return "no_lateral"
    if f.conf_det < CONF_DET_MIN or f.conf_kp_min < CONF_KP_MIN:
        return "keypoints_dudosos"
    return None


def main():
    base = pd.read_csv(config.PROPIO / "fotos.csv", dtype={"animal_id": str})[PAREO]
    ft = pd.read_csv(config.DATA / "propio" / "features_calidad.csv")
    df = base.merge(ft, on="foto", how="left")
    pareada = df.pesaje_id.notna()

    df["motivo_descarte"] = None
    df.loc[~pareada, "motivo_descarte"] = "sin_pesaje"
    df.loc[pareada & ~df.peso_confiable.fillna(False), "motivo_descarte"] = "peso_no_confiable"
    pend = pareada & df.motivo_descarte.isna()
    df.loc[pend, "motivo_descarte"] = [motivo_geometrico(f) for f in df[pend].itertuples()]
    df["lateral_completo"] = df.motivo_descarte.isna()
    df["score"] = (df.conf_det + df.conf_kp_media + df.conf_kp_min) / 3

    # --- hasta N fotos por animal: mejor score primero, separadas en el tiempo ---
    # no usar astype(int64): la unidad del datetime cambia entre versiones de pandas
    df["segs"] = (pd.to_datetime(df.hora) - pd.Timestamp("2026-01-01")).dt.total_seconds() + df.ms / 1000
    for _, g in df[df.lateral_completo].groupby("animal_id"):
        elegidas = []
        for i in g.sort_values("score", ascending=False).index:
            if len(elegidas) < FOTOS_POR_ANIMAL and all(abs(df.at[i, "segs"] - df.at[j, "segs"]) >= SEPARACION_MIN_S for j in elegidas):
                elegidas.append(i)
        df.loc[[i for i in g.index if i not in elegidas], "motivo_descarte"] = "redundante"
    df["uso"] = np.where(df.motivo_descarte.isna(), "incluida", "descartada")

    df = df.rename(columns={"conf_det": "confianza_deteccion"})
    cols = PAREO + ["lateral_completo", "confianza_deteccion", "conf_kp_min", "score", "uso", "motivo_descarte"]
    df[cols].sort_values(["dia", "hora", "foto"]).to_csv(config.PROPIO / "fotos.csv", index=False)

    # --- resumen ---
    inc = df[df.uso == "incluida"]
    print(f"fotos de vaca: {len(df)} | incluidas: {len(inc)} | descartadas: {(df.uso == 'descartada').sum()}")
    print(df[df.uso == "descartada"].motivo_descarte.value_counts().to_string())
    animales = df[pareada].groupby("animal_id").size()
    con_foto = inc.animal_id.nunique()
    print(f"\nanimales con foto asignada: {len(animales)} | con al menos 1 foto incluida: {con_foto} "
          f"| sin ninguna foto usable: {len(animales) - con_foto}")
    print("fotos incluidas por animal:", inc.groupby("animal_id").size().value_counts().sort_index().to_dict())
    print("por dia (animales | fotos):", {d: (g.animal_id.nunique(), len(g)) for d, g in inc.groupby("dia")})
    sin = sorted(set(animales.index) - set(inc.animal_id))
    if sin:
        print(f"animales sin foto usable ({len(sin)}): {sin[:8]}{' ...' if len(sin) > 8 else ''}")


if __name__ == "__main__":
    main()
