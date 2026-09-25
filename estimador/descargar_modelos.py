"""Descarga los modelos entrenados, que no viajan en el repositorio por su tamaño.

Para el que clona el repo, esto es todo lo que hay que correr:

    python descargar_modelos.py

Baja los dos archivos a `modelos/` y verifica que no se hayan corrompido en el camino.
Si la descarga automática falla (sin internet, repo privado), se pueden copiar a mano: ver README.md.

Para quien PUBLICA los modelos: cambiar URL_BASE por la de la release y listo. Si el repositorio es
privado, cada uno necesita `gh auth login` o descargar a mano desde la web.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

# Los .pt se publican como adjuntos de una release de GitHub (ver README.md).
# Si se crea una release nueva (por ejemplo al reentrenar), cambiar el tag del final y los hashes de abajo.
URL_BASE = "https://github.com/luudipietro/Estimador_Bovino_V2/releases/download/modelos-v1"

AQUI = Path(__file__).parent
DESTINO = AQUI / "modelos"
ARCHIVOS = {
    "pose9.pt": "bbe42ed5120f4967c4adad7454d06b856fac19ffc210250e1c766d90b61696f9",
    "peso_convnext.pt": "72011edb80750447308f250874849179a386d95d16558aef6c4f0f49f3ba3244",
}


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def barra(bloques, tam_bloque, total):
    if total > 0:
        pct = min(100, bloques * tam_bloque * 100 // total)
        print(f"\r    {pct}%", end="", flush=True)


def main():
    DESTINO.mkdir(exist_ok=True)
    faltan = []
    for nombre, esperado in ARCHIVOS.items():
        ruta = DESTINO / nombre
        if ruta.exists():
            if sha256(ruta) == esperado:
                print(f"  {nombre}: ya está y es correcto")
                continue
            print(f"  {nombre}: está pero no coincide, se vuelve a bajar")
        faltan.append((nombre, esperado))

    if not faltan:
        print("\nTodo listo. Probar con:  python -c \"from estimador import EstimadorPeso\"")
        return 0

    if URL_BASE.startswith("PENDIENTE"):
        print("\nNo hay URL de descarga configurada todavía.")
        print("Opciones:")
        print("  a) Pedirle los archivos a quien entrenó el modelo y copiarlos en:", DESTINO)
        print("  b) Quien publica: crear la release (ver README.md) y completar URL_BASE en este script.")
        print("\nFaltan:", ", ".join(n for n, _ in faltan))
        return 1

    for nombre, esperado in faltan:
        url = f"{URL_BASE}/{nombre}"
        print(f"  bajando {nombre}...")
        try:
            urllib.request.urlretrieve(url, DESTINO / nombre, reporthook=barra)
            print()
        except Exception as e:
            print(f"\n  ERROR bajando {nombre}: {e}")
            print(f"  Probar a mano: {url}")
            return 1
        real = sha256(DESTINO / nombre)
        if real != esperado:
            print(f"  ERROR: {nombre} se descargó incompleto o corrupto. Borrarlo y reintentar.")
            return 1
        print(f"  {nombre}: descargado y verificado")

    print("\nListo. Probar con:  python -c \"from estimador import EstimadorPeso\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
