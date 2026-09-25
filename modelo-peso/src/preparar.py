"""Descomprime el dataset de Mendeley e inspecciona su estructura real.

Uso:  python src/preparar.py
"""
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def descomprimir():
    if not config.ZIP.exists():
        print(f"FALTA el zip en: {config.ZIP}")
        print(f"Descargalo de {config.URL_DATASET} (boton 'Download All', 2,29 GB)")
        print(f"y dejalo con ese nombre exacto en {config.DATOS}")
        return False

    ya = list(config.CRUDO.rglob("*"))
    if len(ya) > 10:
        print(f"Ya descomprimido: {len(ya)} entradas en {config.CRUDO}")
        return True

    print(f"Descomprimiendo {config.ZIP.name} ...")
    with zipfile.ZipFile(config.ZIP) as z:
        z.extractall(config.CRUDO)
    print("Listo.")
    return True


def inspeccionar():
    archivos = [p for p in config.CRUDO.rglob("*") if p.is_file()]
    print(f"\n{len(archivos)} archivos en total\n")

    print("--- por extension ---")
    for ext, n in Counter(p.suffix.lower() for p in archivos).most_common():
        tam = sum(p.stat().st_size for p in archivos if p.suffix.lower() == ext)
        print(f"  {ext or '(sin ext)':10s}  {n:5d}  {tam/1e6:9.1f} MB")

    print("\n--- arbol de carpetas ---")
    for d in sorted({p.parent for p in archivos}):
        rel = d.relative_to(config.CRUDO)
        n = sum(1 for p in archivos if p.parent == d)
        print(f"  {rel if str(rel) != '.' else '(raiz)'}  ->  {n} archivos")

    print("\n--- primeros nombres de imagen ---")
    imgs = [p for p in archivos if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    for p in sorted(imgs)[:12]:
        print(f"  {p.relative_to(config.CRUDO)}")
    if len(imgs) > 12:
        print(f"  ... y {len(imgs)-12} mas")

    print("\n--- archivos de anotaciones ---")
    tablas = [p for p in archivos if p.suffix.lower() in {".xlsx", ".xls", ".csv", ".txt", ".json"}]
    for p in tablas:
        print(f"\n  === {p.relative_to(config.CRUDO)} ({p.stat().st_size/1e3:.1f} KB) ===")
        volcar(p)

    return archivos


def volcar(p: Path, filas=8):
    try:
        if p.suffix.lower() in {".xlsx", ".xls"}:
            import openpyxl
            wb = openpyxl.load_workbook(p, data_only=True)
            for ws in wb.worksheets:
                print(f"  hoja '{ws.title}' dims={ws.dimensions}")
                for i, fila in enumerate(ws.iter_rows(values_only=True)):
                    if i >= filas:
                        print("    ...")
                        break
                    print("   ", " | ".join("" if c is None else str(c) for c in fila))
        else:
            with open(p, encoding="utf-8", errors="replace") as f:
                for i, linea in enumerate(f):
                    if i >= filas:
                        print("    ...")
                        break
                    print("   ", linea.rstrip()[:200])
    except Exception as e:
        print(f"    (no se pudo leer: {type(e).__name__}: {e})")


if __name__ == "__main__":
    if descomprimir():
        inspeccionar()
