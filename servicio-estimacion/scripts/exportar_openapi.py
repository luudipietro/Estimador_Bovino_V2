"""Genera (o comprueba) `servicio-estimacion/openapi.json`, el contrato que lee el equipo del adaptador C#.

    python scripts/exportar_openapi.py               # escribe openapi.json
    python scripts/exportar_openapi.py --comprobar   # exit 1 si el archivo no es el que genera la app

No arranca el servicio ni carga el modelo: arma la app y le pide su esquema.
"""

import argparse
import sys
from pathlib import Path

from bv_estimacion.api.openapi import esquema_openapi, serializar_esquema

ARCHIVO = Path(__file__).resolve().parent.parent / "openapi.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--salida", type=Path, default=ARCHIVO, help="Dónde escribir o comprobar (por defecto openapi.json)"
    )
    parser.add_argument("--comprobar", action="store_true", help="No escribe: falla si el archivo está desactualizado")
    argumentos = parser.parse_args()

    generado = serializar_esquema(esquema_openapi())

    if argumentos.comprobar:
        actual = argumentos.salida.read_text(encoding="utf-8") if argumentos.salida.is_file() else None
        if actual != generado:
            print(
                f"{argumentos.salida.name} no coincide con la API actual. "
                "Regenerar con: python scripts/exportar_openapi.py",
                file=sys.stderr,
            )
            return 1
        print(f"{argumentos.salida.name} está al día.")
        return 0

    argumentos.salida.write_text(generado, encoding="utf-8", newline="\n")
    print(f"Escrito {argumentos.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
