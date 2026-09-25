from pathlib import Path

# El dataset vive DENTRO del repo, en data/, pero se ignora en git (ver .gitignore):
# cada persona que clona el repo lo descarga a mano y lo pone en esta misma carpeta.
# La ruta es relativa a este archivo, no al usuario ni a la maquina, para que le
# funcione igual a cualquiera del equipo sin tocar nada.
DATOS = Path(__file__).resolve().parent / "data"

ZIP = DATOS / "h2s22wr5py-2.zip"
CRUDO = DATOS / "crudo"
MASCARAS = DATOS / "mascaras"
TABLAS = DATOS / "tablas"

URL_DATASET = "https://data.mendeley.com/datasets/h2s22wr5py/2"

for d in (DATOS, CRUDO, MASCARAS, TABLAS):
    d.mkdir(parents=True, exist_ok=True)

# Segundo dataset (externo, validacion del marcador de referencia). Vive en la raiz
# del proyecto, afuera de modelo-peso/, no versionado. Ver CLAUDE.md.
DATASET_ACMEAI = Path(__file__).resolve().parent.parent / "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV"
