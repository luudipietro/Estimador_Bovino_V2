"""Dataset propio, etapa 1b: controles de consistencia sobre `dataset_propio/pesajes_leidos.csv`.

Las lecturas de pantalla se hicieron a ojo, así que se verifican con redundancia que trae la propia pantalla:
  1. formato: peso 150-700 kg, IDE de 15 dígitos (`NNN NNNNNNNNNNNN`)
  2. IDE repetido dentro de un mismo día (pesaje doble o lectura equivocada)
  3. cuando el animal es nuevo en el sistema (total_dias == dias): GPV total == GDM * dias
  4. animales presentes los DOS días: la diferencia de peso debe coincidir con la diferencia de GPV total
     (ambos cuentan desde el mismo peso inicial), lo que valida a la vez el IDE y el peso de ambas lecturas
Uso:  uv run python 40b_pesajes_controles.py
"""
import re

import pandas as pd

import config

IDE_OK = re.compile(r"^\d{3} \d{12}$")


def main():
    df = pd.read_csv(config.PROPIO / "pesajes_leidos.csv", dtype={"ide": str})
    df["dia"] = df.carpeta.str.split("pesos").str[0]
    pes = df[df.estado != "no_es_pesaje"].copy()
    print(f"{len(df)} filas leídas | {len(pes)} pesajes | por día: {pes.groupby('dia').size().to_dict()}")
    print(f"estados: {pes.estado.value_counts().to_dict()}")

    fuera = pes[(pes.peso_kg < 150) | (pes.peso_kg > 700)]
    print(f"\n[1] pesos fuera de 150-700 kg: {len(fuera)}")
    sin_ide = pes[pes.ide.isna()]
    print(f"    sin IDE: {len(sin_ide)} -> {sin_ide.archivo.tolist()}")
    mal = pes[pes.ide.notna() & ~pes.ide.fillna('').str.match(IDE_OK)]
    print(f"    IDE con formato raro: {len(mal)} {mal[['archivo', 'ide']].values.tolist()}")

    con_ide = pes[pes.ide.notna()]
    dup = con_ide[con_ide.duplicated(["dia", "ide"], keep=False)].sort_values(["dia", "ide"])
    print(f"\n[2] IDE repetido dentro del mismo día: {len(dup)}")
    if len(dup):
        print(dup[["archivo", "peso_kg", "ide", "estado"]].to_string(index=False))

    nuevos = pes[pes.gdm.notna() & pes.gpv_total.notna() & (pes.total_dias == pes.dias)]
    err = (nuevos.gdm * nuevos.dias - nuevos.gpv_total).abs()
    malos = nuevos[err > 1.5]
    print(f"\n[3] animales nuevos (total_dias == dias): {len(nuevos)}; GPV != GDM*dias (tol 1,5 kg): {len(malos)}")
    if len(malos):
        print(malos[["archivo", "peso_kg", "gdm", "gpv_total", "dias"]].to_string(index=False))

    a, b = con_ide[con_ide.dia == "8-10"], con_ide[con_ide.dia == "22-10"]
    comun = a.merge(b, on="ide", suffixes=("_8", "_22"))
    print(f"\n[4] animales en ambos días (mismo IDE): {len(comun)}")
    if len(comun):
        comun["d_peso"] = comun.peso_kg_22 - comun.peso_kg_8
        comun["d_gpv"] = comun.gpv_total_22 - comun.gpv_total_8
        comun["dif"] = (comun.d_peso - comun.d_gpv).abs()
        print(comun[["ide", "archivo_8", "archivo_22", "peso_kg_8", "peso_kg_22", "d_peso", "d_gpv", "dif"]]
              .to_string(index=False))

    print("\nResumen de peso por día (kg):")
    print(pes.groupby("dia").peso_kg.describe().round(1).to_string())


if __name__ == "__main__":
    main()
