"""Dataset propio, etapa 2: une cada foto de vaca con su pesaje por la HORA de la foto.

En el campo trabajaron dos teléfonos en paralelo: uno fotografiaba la pantalla de la balanza (peso + caravana) y el
otro a la vaca saliendo de la manga. El ciclo fue siempre: foto de pantalla -> la vaca sale -> fotos de la vaca ->
siguiente pantalla. Por eso las fotos de vaca con  t_pantalla_i <= t < t_pantalla_(i+1)  son del animal del pesaje i.
Los relojes de los dos teléfonos están sincronizados (~7-9 s entre pantalla y primera foto de vaca).

Entrada : dataset_propio/pesajes_leidos.csv (lectura de pantallas, etapa 1) + las fotos originales.
Salida  : dataset_propio/pesajes.csv  (1 fila por pesaje real, con pesaje_id, hora y n_fotos_vaca)
          dataset_propio/fotos.csv    (1 fila por foto de vaca: pesaje, animal, peso, teléfono, resolución...)
La calidad de cada foto la agrega 42_propio_calidad.py sobre fotos.csv.
Uso:  uv run python 41_propio_parear.py
"""
from datetime import datetime

import pandas as pd
import pillow_heif
from PIL import Image

import config

pillow_heif.register_heif_opener()

DIAS = {"8-10": "20261008", "22-10": "20261022"}
VENTANA_MAX_S = 60          # una foto de vaca a más de esto de la pantalla anterior no se asigna a nadie
EXT_FOTO = {".jpg", ".jpeg", ".heic"}
ORIENT_GIRA = {5, 6, 7, 8}  # orientaciones EXIF que intercambian ancho y alto
EXIF_FOCAL_35MM = 0xA405


def leer_foto(ruta):
    """Hora (DateTimeOriginal), teléfono, ancho, alto (ya con la orientación aplicada) y focal 35 mm."""
    with Image.open(ruta) as im:
        ex = im.getexif()
        sub = ex.get_ifd(0x8769)
        hora = datetime.strptime(sub[0x9003], "%Y:%m:%d %H:%M:%S")
        ancho, alto = im.size
        if ex.get(0x0112) in ORIENT_GIRA and ruta.suffix.lower() != ".heic":
            ancho, alto = alto, ancho   # pillow-heif ya devuelve el tamaño rotado; el JPG no
        subseg = sub.get(0x9291) or "0"
        return {"hora": hora, "ms": int(str(subseg).ljust(3, "0")[:3]), "telefono": str(ex.get(0x0110, "")).strip(),
                "ancho": ancho, "alto": alto, "focal_35mm": sub.get(EXIF_FOCAL_35MM)}


def main():
    leidos = pd.read_csv(config.PROPIO / "pesajes_leidos.csv", dtype={"ide": str})
    leidos["dia"] = leidos.carpeta.str.split("pesos").str[0]

    # --- pantallas: hora EXIF y pesaje_id correlativo por día (las que no son pesaje solo marcan un corte) ---
    filas = []
    for r in leidos.itertuples():
        info = leer_foto(config.PROPIO / r.carpeta / r.archivo)
        filas.append({**r._asdict(), "hora_pantalla": info["hora"], "telefono_pantalla": info["telefono"]})
    pant = pd.DataFrame(filas).drop(columns="Index").sort_values(["dia", "hora_pantalla", "archivo"]).reset_index(drop=True)
    pant["pesaje_id"] = None
    for dia, g in pant[pant.estado != "no_es_pesaje"].groupby("dia"):
        pant.loc[g.index, "pesaje_id"] = [f"{DIAS[dia]}_{i:03d}" for i in range(1, len(g) + 1)]
    pant["animal_id"] = pant.ide.fillna("").str.replace(" ", "")
    sin_ide = pant.pesaje_id.notna() & (pant.animal_id == "")
    pant.loc[sin_ide, "animal_id"] = "SINIDE_" + pant.loc[sin_ide, "pesaje_id"]
    pant["peso_confiable"] = pant.estado.isin(["registrado", "estable_sin_registrar"])

    # --- fotos de vaca ---
    fotos, video = [], []
    for dia in DIAS:
        carpeta = config.PROPIO / f"{dia}vacas"
        for p in sorted(carpeta.iterdir()):
            if p.suffix.lower() not in EXT_FOTO:
                video.append(p.name)
                continue
            info = leer_foto(p)
            fotos.append({"foto": f"{dia}vacas/{p.name}", "dia": dia, **info})
    fv = pd.DataFrame(fotos)

    # --- pareo: la última pantalla anterior (o de la misma hora) a cada foto de vaca ---
    fv["pesaje_id"], fv["dt_s"], fv["motivo"] = None, None, None
    for dia in DIAS:
        cortes = pant[pant.dia == dia][["hora_pantalla", "pesaje_id"]].sort_values("hora_pantalla")
        idx = fv.index[fv.dia == dia]
        sub = fv.loc[idx].sort_values(["hora", "ms"])
        m = pd.merge_asof(sub[["hora"]].astype("datetime64[us]").reset_index(), cortes.astype({"hora_pantalla": "datetime64[us]"}),
                          left_on="hora", right_on="hora_pantalla", direction="backward").set_index("index")
        dt = (m.hora - m.hora_pantalla).dt.total_seconds()
        fv.loc[m.index, "pesaje_id"] = m.pesaje_id
        fv.loc[m.index, "dt_s"] = dt
        fv.loc[m.index[m.hora_pantalla.isna()], "motivo"] = "sin_pesaje_previo"
        fv.loc[m.index[dt > VENTANA_MAX_S], "motivo"] = "lejos_de_la_pantalla"
        # la pantalla anterior era un corte que no es pesaje (foto del equipo): no hay animal asociado
        fv.loc[m.index[m.pesaje_id.isna() & m.hora_pantalla.notna()], "motivo"] = "pantalla_no_es_pesaje"
    fv.loc[fv.motivo.notna(), "pesaje_id"] = None

    fv = fv.merge(pant[["pesaje_id", "animal_id", "peso_kg", "peso_confiable"]], on="pesaje_id", how="left")
    fv["uso"] = fv.motivo.isna().map({True: "candidata", False: "descartada"})
    fv = fv.rename(columns={"motivo": "motivo_descarte"})
    cols = ["foto", "dia", "hora", "ms", "pesaje_id", "animal_id", "peso_kg", "peso_confiable", "dt_s", "telefono",
            "ancho", "alto", "focal_35mm", "uso", "motivo_descarte"]
    fv = fv.sort_values(["dia", "hora", "foto"])[cols]

    # --- pesajes.csv ---
    n = fv[fv.uso == "candidata"].groupby("pesaje_id").size()
    pes = pant[pant.pesaje_id.notna()].copy()
    pes["n_fotos_vaca"] = pes.pesaje_id.map(n).fillna(0).astype(int)
    pes = pes.rename(columns={"archivo": "foto_pantalla", "obs": "observaciones"})
    pes = pes[["pesaje_id", "dia", "hora_pantalla", "foto_pantalla", "peso_kg", "ide", "animal_id", "estado", "peso_confiable",
               "corral", "sexo", "gdm", "gpv_total", "total_dias", "dias", "telefono_pantalla", "n_fotos_vaca", "observaciones"]]
    pes = pes.rename(columns={"estado": "estado_pantalla"})

    pes.to_csv(config.PROPIO / "pesajes.csv", index=False)
    fv.to_csv(config.PROPIO / "fotos.csv", index=False)

    # --- resumen ---
    print(f"pesajes: {len(pes)} | fotos de vaca: {len(fv)} (+ {len(video)} video fuera: {video})")
    print(f"candidatas: {(fv.uso == 'candidata').sum()} | descartadas: {fv.motivo_descarte.value_counts().to_dict()}")
    print(f"fotos por pesaje (candidatas): {pes.groupby('dia').n_fotos_vaca.describe().round(1).to_dict('index')}")
    print(f"pesajes sin ninguna foto: {pes[pes.n_fotos_vaca == 0][['pesaje_id', 'foto_pantalla']].values.tolist()}")
    c = fv[fv.uso == "candidata"]
    print(f"demora foto de vaca desde la pantalla (s): min {c.dt_s.min():.0f} | mediana {c.dt_s.median():.0f} | max {c.dt_s.max():.0f}")
    assert fv.foto.is_unique, "foto repetida"


if __name__ == "__main__":
    main()
