"""Dataset de prueba, etapa 3: calidad de cada foto, confianza del pareo y armado de `dataset_prueba/curado/`.

Reutiliza los criterios del dataset propio (42a/42) para MARCAR cada foto (`lateral_completo`: un animal dominante, cuerpo
completo, vista lateral, keypoints confiables), pero **no descarta por eso**: en un test, que el modelo diga "no sé" ante una
foto mala es parte de lo que se mide. Lo que sí decide es **qué pesajes son seguros para evaluar**, según las ráfagas que
tienen un animal (`hay_animal`: un animal dominante, detectado y no muy chico):

  confianza_pareo = alta      -> exactamente UNA ráfaga con animal llega antes de la pantalla y la demora es normal
                  = ambigua   -> dos o más ráfagas con animal compiten por la misma pantalla (no se sabe cuál es la vaca pesada)
                  = sin_foto  -> ninguna ráfaga con animal llega antes de esa pantalla
                  = duplicado -> es el re-pesaje de otro pesaje: no tiene fotos propias

Solo los de confianza **alta** entran a `curado/` y se usan para medir. Los demás quedan registrados con su motivo: un test
más chico pero limpio vale más que uno más grande con fotos de otro animal (un par equivocado suma error sin que se note).

Uso:  uv run --extra cpu python 52_prueba_calidad.py
"""

import importlib
import time

import numpy as np
import pandas as pd

import config

feat = importlib.import_module("42a_propio_features")  # escena(), abrir_orientada(), EstimadorPeso del módulo
cal = importlib.import_module("42_propio_calidad")  # motivo_geometrico() y sus umbrales
cur = importlib.import_module("43_propio_curar")  # reducir() a 1024 px

FOTOS_POR_ANIMAL = 3
SEPARACION_MIN_S = 0.35
LAG_MAX_S = 70.0  # una ráfaga que precede a la pantalla en más de esto ya no se considera su foto
LAG_FIN_MIN_S = -7.0  # tolerancia entre relojes: la ráfaga no puede terminar mucho DESPUÉS de la pantalla
CACHE = config.DATA / "prueba" / "features_calidad.csv"


def medir(fotos):
    """Corre detector + pose (los del módulo `estimador`) sobre cada foto. Cacheado: retoma donde quedó."""
    hechas = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["foto"])
    pendientes = [f for f in fotos if f not in set(hechas.foto)]
    if pendientes:
        est = feat.EstimadorPeso(config.ROOT.parent / "estimador" / "modelos")
        filas, t0 = [], time.time()
        for i, f in enumerate(pendientes, 1):
            im = feat.abrir_orientada(config.PRUEBA / f)
            fila = {"foto": f, **feat.escena(est, im)}
            r = est.estimar(im)
            fila.update({"conf_det": r.get("confianza_deteccion"), "conf_kp_min": r.get("confianza_keypoint_minima")})
            if r.get("keypoints"):
                a = np.array(list(r["keypoints"].values()))
                fila["conf_kp_media"] = float(a[:, 2].mean())
                fila["aspecto_kp"] = float((a[:, 0].max() - a[:, 0].min()) / max(a[:, 1].max() - a[:, 1].min(), 1))
            filas.append(fila)
            if i % 25 == 0 or i == len(pendientes):
                print(f"  {i}/{len(pendientes)}  ({(time.time() - t0) / i:.1f} s/foto)", flush=True)
        hechas = pd.concat([hechas, pd.DataFrame(filas)], ignore_index=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        hechas.to_csv(CACHE, index=False)
    return hechas


def main():
    pes = pd.read_csv(config.PRUEBA / "pesajes.csv", dtype={"ide": str, "animal_id": str}, keep_default_na=False)
    fot = pd.read_csv(config.PRUEBA / "fotos.csv", dtype={"animal_id": str}, keep_default_na=False)
    # idempotente: solo las columnas que escribe 51_prueba_parear.py (esta etapa agrega las suyas cada vez)
    fot = fot[["foto", "dia", "hora", "rafaga", "pesaje_id", "animal_id", "peso_kg", "lag_s", "telefono", "ancho", "alto"]]
    pes = pes[["pesaje_id", "dia", "hora_pantalla", "foto_pantalla", "peso_kg", "peso_animal_kg", "ide", "animal_id",
               "estado_pantalla", "corral", "sexo", "gdm", "gpv_total", "total_dias", "dias", "duplicado_de",
               "n_rafagas_candidatas", "n_fotos_candidatas", "observaciones"]]
    ft = medir(list(fot.foto))
    df = fot.merge(ft, on="foto", how="left")

    # --- calidad por foto (mismos criterios que el dataset propio) ---
    df["calidad"] = [cal.motivo_geometrico(f) if pd.notna(f.conf_det) else "sin_deteccion" for f in df.itertuples()]
    df["calidad"] = df.calidad.fillna("ok")  # "ok" = lateral, cuerpo completo y keypoints confiables
    df["lateral_completo"] = df.calidad == "ok"
    df["score"] = (df.conf_det + df.conf_kp_media + df.conf_kp_min) / 3

    # --- confianza del pareo por pesaje, a nivel de ráfaga ---
    df["hay_animal"] = ~df.calidad.isin(["sin_deteccion", "varios_animales", "muy_chica"]) & (df.pesaje_id != "")
    usables = df[df.hay_animal].groupby(["pesaje_id", "rafaga"]).agg(
        n=("foto", "size"), lag_ini=("lag_s", "max"), lag_fin=("lag_s", "min")).reset_index()
    usables = usables[(usables.lag_ini <= LAG_MAX_S) & (usables.lag_fin >= LAG_FIN_MIN_S)]
    por_pesaje = usables.groupby("pesaje_id").rafaga.nunique()
    pes["rafagas_usables"] = pes.pesaje_id.map(por_pesaje).fillna(0).astype(int)

    def confianza(p):
        if p.duplicado_de:
            return "duplicado"
        return {0: "sin_foto", 1: "alta"}.get(p.rafagas_usables, "ambigua")

    pes["confianza_pareo"] = [confianza(p) for p in pes.itertuples()]

    # --- qué fotos se usan: solo las de pesajes con pareo de confianza alta, hasta 3 por animal ---
    alta = set(pes[pes.confianza_pareo == "alta"].pesaje_id)
    rafaga_ok = {r.pesaje_id: r.rafaga for r in usables.itertuples()}
    df["uso"] = "descartada"
    df["segs"] = (df.hora.map(pd.Timestamp.fromisoformat) - pd.Timestamp("2026-01-01")).dt.total_seconds()
    for pid in alta:
        g = df[(df.pesaje_id == pid) & df.hay_animal & (df.rafaga == rafaga_ok[pid])]
        elegidas = []
        # primero las laterales completas; si no hay, las de mayor confianza (el modelo dirá "no sé" y eso también se mide)
        for i in g.sort_values(["lateral_completo", "score"], ascending=False).index:
            if len(elegidas) < FOTOS_POR_ANIMAL and all(abs(df.at[i, "segs"] - df.at[j, "segs"]) >= SEPARACION_MIN_S for j in elegidas):
                elegidas.append(i)
        df.loc[elegidas, "uso"] = "incluida"
    # por qué no se usa cada foto descartada: primero lo que dice la foto, después lo que dice el pareo
    conf_de = pes.set_index("pesaje_id").confianza_pareo
    motivos = []
    for f in df.itertuples():
        if f.uso == "incluida":
            motivos.append("")
        elif f.pesaje_id == "":
            motivos.append("sin_pesaje")
        elif not f.hay_animal:
            motivos.append(f.calidad)  # sin_deteccion, varios_animales o muy_chica
        elif conf_de[f.pesaje_id] != "alta":
            motivos.append(f"pareo_{conf_de[f.pesaje_id]}")
        elif f.rafaga != rafaga_ok[f.pesaje_id]:
            motivos.append("otra_rafaga")
        else:
            motivos.append("redundante")
    df["motivo_descarte"] = motivos

    # --- curado/: las fotos incluidas, a 1024 px, sin split (todo es test) ---
    df["archivo_curado"] = ""
    for animal, g in df[df.uso == "incluida"].groupby("animal_id"):
        for n, i in enumerate(g.sort_values("hora").index, 1):
            rel = f"curado/{animal}__{df.at[i, 'pesaje_id']}__{n}.jpg"
            if not (config.PRUEBA / rel).exists():
                cur.reducir(config.PRUEBA / df.at[i, "foto"], config.PRUEBA / rel)
            df.at[i, "archivo_curado"] = rel
    en_disco = {p.relative_to(config.PRUEBA).as_posix() for p in (config.PRUEBA / "curado").glob("*.jpg")}
    for sobra in en_disco - set(df.archivo_curado):
        (config.PRUEBA / sobra).unlink()

    pes["n_fotos_incluidas"] = pes.pesaje_id.map(df[df.uso == "incluida"].groupby("pesaje_id").size()).fillna(0).astype(int)
    cols = ["foto", "dia", "hora", "rafaga", "pesaje_id", "animal_id", "peso_kg", "lag_s", "telefono", "ancho", "alto",
            "calidad", "lateral_completo", "hay_animal", "conf_det", "conf_kp_min", "score", "uso", "motivo_descarte", "archivo_curado"]
    df[cols].to_csv(config.PRUEBA / "fotos.csv", index=False)
    inc = df[df.uso == "incluida"]
    # para `estimador/probar.py --carpeta dataset_prueba/curado --pesos dataset_prueba/pesos_curado.csv`
    pd.DataFrame({"foto": inc.archivo_curado.str.split("/").str[-1], "peso_kg": inc.peso_kg, "animal": inc.animal_id,
                  "lateral_completo": inc.lateral_completo}).to_csv(config.PRUEBA / "pesos_curado.csv", index=False)
    pes.to_csv(config.PRUEBA / "pesajes.csv", index=False)

    # --- resumen ---
    print(f"\nfotos de vaca: {len(df)} | incluidas: {(df.uso == 'incluida').sum()} | descartadas: {(df.uso == 'descartada').sum()}")
    print(df[df.uso == "descartada"].motivo_descarte.value_counts().to_string())
    print("\nconfianza del pareo por pesaje:", pes.confianza_pareo.value_counts().to_dict())
    ev = pes[pes.n_fotos_incluidas > 0]
    print(f"animales evaluables: {len(ev)} | fotos: {int(ev.n_fotos_incluidas.sum())} | peso {ev.peso_animal_kg.min():.0f}-{ev.peso_animal_kg.max():.0f} kg"
          f" (mediana {ev.peso_animal_kg.median():.0f}) | machos con IDE: {(ev.sexo == 'Macho').sum()} | sin IDE: {ev.ide.eq('').sum()}")

    # --- integridad: ¿algún IDE está también en el dataset propio? (sería fuga hacia el entrenamiento) ---
    propio = pd.read_csv(config.PROPIO / "pesajes.csv", dtype={"animal_id": str}, keep_default_na=False)
    comunes = sorted(set(pes[pes.ide != ""].animal_id) & set(propio.animal_id))
    print(f"IDE en común con el dataset propio (fuga): {len(comunes)} {comunes}")
    sin_ide = pes[(pes.ide == "") & (pes.n_fotos_incluidas > 0)]
    print(f"animales sin IDE evaluables ({len(sin_ide)}): no se puede descartar que alguno esté también en el dataset propio")


if __name__ == "__main__":
    main()
