"""Reloj, espera e IDs del sistema real. Los tests de los casos de uso usan fakes deterministas."""

import os
import secrets
import threading
import time
from datetime import UTC, datetime

import anyio

from bv_estimacion.dominio.valores import IdTrabajo

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class RelojSistema:
    def ahora(self) -> datetime:
        return datetime.now(UTC)


class DormidorAnyio:
    async def dormir(self, segundos: float) -> None:
        await anyio.sleep(segundos)


def _codificar(valor: int, largo: int) -> str:
    caracteres = []
    for _ in range(largo):
        caracteres.append(_CROCKFORD[valor & 31])
        valor >>= 5
    return "".join(reversed(caracteres))


class GeneradorIdsUlid:
    """IDs `trb_<ULID>`: 26 caracteres, ordenables por tiempo (48 bits de milisegundos y 80 de azar).

    Dentro de un mismo milisegundo el azar se incrementa, así que los IDs salen siempre en orden creciente.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ultimo_ms = -1
        self._ultimo_azar = 0

    def nuevo_id_trabajo(self) -> IdTrabajo:
        with self._lock:
            ahora_ms = time.time_ns() // 1_000_000
            if ahora_ms > self._ultimo_ms:
                self._ultimo_ms = ahora_ms
                self._ultimo_azar = int.from_bytes(os.urandom(10), "big")
            else:
                # mismo milisegundo (o el reloj retrocedió): se sigue desde el último para no repetir ni desordenar
                self._ultimo_azar += 1
                if self._ultimo_azar >= 1 << 80:
                    self._ultimo_ms += 1
                    self._ultimo_azar = secrets.randbits(80)
            return IdTrabajo(f"trb_{_codificar(self._ultimo_ms, 10)}{_codificar(self._ultimo_azar, 16)}")
