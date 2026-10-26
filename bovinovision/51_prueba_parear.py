"""Dataset de prueba, etapa 2: une las fotos de vaca con los pesajes de la jornada del 6/10 (fotos de vaca ANTES del pesaje).

A diferencia de las jornadas del 8/10 y 22/10 (pantalla -> la vaca sale), acá las vacas se fotografiaron **antes** de
pasar por la balanza: las fotos de un animal quedan ANTES de la foto de su pantalla. Regla: cada ráfaga de fotos de vaca
(fotos separadas por menos de GAP_RAFAGA_S) se asigna al **primer pesaje que ocurre después de que empieza la ráfaga**.

Esta regla sola no alcanza para asegurar el pareo (a veces hay dos ráfagas antes de una pantalla, o ninguna). Acá solo se
arma el candidato y se registran los datos necesarios (ráfaga, demora); la decisión de cuáles pares son confiables la toma
`52_prueba_calidad.py`, que ya sabe qué ráfagas tienen un animal usable.

Entrada: dataset_prueba/pesajes_leidos.csv (lectura de pantallas) y las fotos originales.
Salida : dataset_prueba/pesajes.csv y dataset_prueba/fotos.csv
Uso:  uv run --extra cpu python 51_prueba_parear.py
"""

import pandas as pd
import pillow_heif
from PIL import Image

import config

pillow_heif.register_heif_opener()

DIA = "20261006"
CARPETA_PANTALLAS = "martes_6_10_26"
CARPETA_VACAS = "martes6-10Vacas"
GAP_RAFAGA_S = 6.0  # entre dos fotos de la misma ráfaga; más que esto empieza otra (otro animal u otro momento)
ORIENT_GIRA = {5, 6, 7, 8}


def leer_exif(ruta):
    with Image.open(ruta) as im:
        ex = im.getexif()
        sub = ex.get_ifd(0x8769)
        ancho, alto = im.size
        if ex.get(0x0112) in ORIENT_GIRA:
            ancho, alto = alto, ancho
        hora = pd.Timestamp(sub[0x9003].replace(":", "-", 2))
        ms = int(str(sub.get(0x9291, "0")).ljust(3, "0")[:3])
        return {"t": hora.timestamp() + ms / 1000, "hora": hora, "telefono": str(ex.get(0x0110, "")).strip(),
                "ancho": ancho, "alto": alto}


def main():
    leidos = pd.read_csv(config.PRUEBA / "pesajes_leidos.csv", dtype={"ide": str})
    pant = pd.DataFrame([{**r._asdict(), **leer_exif(config.PRUEBA / CARPETA_PANTALLAS / r.archivo)}
                         for r in leidos.itertuples(index=False)]).sort_values("t").reset_index(drop=True)
    pant["pesaje_id"] = [f"{DIA}_{i:03d}" for i in range(1, len(pant) + 1)]
    pant["ide"] = pant.ide.fillna("")
    pant["animal_id"] = pant.ide.str.replace(" ", "")
    pant.loc[pant.animal_id == "", "animal_id"] = "SINIDE_" + pant.pesaje_id
    # re-pesaje: mismo IDE en dos pantallas consecutivas casi seguidas -> es un solo animal pesado dos veces
    pant["duplicado_de"] = ""
    for i in range(1, len(pant)):
        a, b = pant.loc[i - 1], pant.loc[i]
        if b.ide and b.ide == a.ide and abs(b.peso_kg - a.peso_kg) <= 3 and b.t - a.t < 30:
            pant.loc[i, "duplicado_de"] = a.pesaje_id
    # el animal tiene un solo peso: el promedio de sus dos lecturas
    for i in pant.index[pant.duplicado_de != ""]:
        promedio = pant.loc[[i, i - 1], "peso_kg"].mean()
        pant.loc[[i, i - 1], "peso_animal_kg"] = promedio
    pant["peso_animal_kg"] = pant.peso_animal_kg.fillna(pant.peso_kg)

    fotos = pd.DataFrame([{"foto": f"{CARPETA_VACAS}/{p.name}", **leer_exif(p)}
                          for p in sorted((config.PRUEBA / CARPETA_VACAS).glob("*.JPG"))]).sort_values("t").reset_index(drop=True)
    fotos["rafaga"] = (fotos.t.diff() > GAP_RAFAGA_S).cumsum()

    # cada ráfaga -> el primer pesaje (que no sea un re-pesaje) que ocurre después de que empieza
    destinos = pant[pant.duplicado_de == ""]
    inicio = fotos.groupby("rafaga").t.min()
    destino = {}
    for rafaga, t0 in inicio.items():
        siguiente = destinos[destinos.t >= t0]
        destino[rafaga] = siguiente.pesaje_id.iloc[0] if len(siguiente) else ""
    fotos["pesaje_id"] = fotos.rafaga.map(destino)
    t_pantalla = pant.set_index("pesaje_id").t
    fotos["lag_s"] = (fotos.pesaje_id.map(t_pantalla) - fotos.t).round(1)  # segundos que la foto precede a la pantalla
    fotos = fotos.merge(pant[["pesaje_id", "animal_id", "peso_animal_kg"]].rename(columns={"peso_animal_kg": "peso_kg"}),
                        on="pesaje_id", how="left")

    pant["n_rafagas_candidatas"] = pant.pesaje_id.map(fotos.groupby("pesaje_id").rafaga.nunique()).fillna(0).astype(int)
    pant["n_fotos_candidatas"] = pant.pesaje_id.map(fotos.groupby("pesaje_id").size()).fillna(0).astype(int)
    pant = pant.rename(columns={"archivo": "foto_pantalla", "estado": "estado_pantalla", "obs": "observaciones", "hora": "hora_pantalla"})
    pant["dia"] = "6-10"
    pant[["pesaje_id", "dia", "hora_pantalla", "foto_pantalla", "peso_kg", "peso_animal_kg", "ide", "animal_id", "estado_pantalla",
          "corral", "sexo", "gdm", "gpv_total", "total_dias", "dias", "duplicado_de", "n_rafagas_candidatas",
          "n_fotos_candidatas", "observaciones"]].to_csv(config.PRUEBA / "pesajes.csv", index=False)
    fotos["dia"] = "6-10"
    fotos[["foto", "dia", "hora", "rafaga", "pesaje_id", "animal_id", "peso_kg", "lag_s", "telefono", "ancho", "alto"]].to_csv(
        config.PRUEBA / "fotos.csv", index=False)

    print(f"{len(pant)} pantallas ({(pant.duplicado_de != '').sum()} re-pesaje) | {len(fotos)} fotos de vaca en {fotos.rafaga.nunique()} ráfagas")
    print(f"pesajes sin ninguna ráfaga candidata : {int((pant.n_rafagas_candidatas == 0).sum())}")
    print(f"pesajes con una ráfaga               : {int((pant.n_rafagas_candidatas == 1).sum())}")
    print(f"pesajes con 2 o más ráfagas          : {int((pant.n_rafagas_candidatas >= 2).sum())}")
    print(f"animales distintos (por IDE o sin IDE): {pant[pant.duplicado_de == ''].animal_id.nunique()}")
    huerfanas = fotos[fotos.pesaje_id == ""]
    if len(huerfanas):
        print(f"fotos posteriores al último pesaje: {len(huerfanas)}")


if __name__ == "__main__":
    main()
