"""Rutas y parámetros globales. Todo relativo a este archivo: funciona igual en Windows y Fedora."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TABLAS = DATA / "tablas"
CACHE = DATA / "cache"
RUNS = ROOT / "runs"

# Dataset AcmeAI/BMGF: vive fuera del repo, al lado de bovinovision/ (sobreescribible con ACMEAI_DIR)
ACMEAI = Path(os.environ.get(
    "ACMEAI_DIR",
    ROOT.parent / "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV",
))

# Dataset propio (campo, 8/10 y 22/10 de 2026): vive al lado de bovinovision/, las fotos no se versionan
PROPIO = Path(os.environ.get("PROPIO_DIR", ROOT.parent / "dataset_propio"))

SEED = 42
SPLIT_FRACCIONES = {"train": 0.70, "val": 0.15, "test": 0.15}

# Bandas de peso (kg) para reportar el error por rango; siempre junto al baseline de la mediana
BANDAS_PESO = [0, 100, 200, 300, 10_000]

for _d in (TABLAS, CACHE, RUNS):
    _d.mkdir(parents=True, exist_ok=True)
