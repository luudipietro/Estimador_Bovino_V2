"""Almacén temporal de imágenes en disco (implementa `AlmacenTemporalImagenes`).

Va en disco y no en memoria: con 4 fotos de hasta 25 MB por pedido y varios pedidos en cola, la RAM se llena. Las
imágenes se borran al terminar el trabajo (`ProcesarEstimacion`); el TTL es el respaldo para lo que quede
huérfano si el proceso se cae a mitad de camino.
"""

import re
import secrets
import time
from collections.abc import Callable
from pathlib import Path

import anyio

_REFERENCIA = re.compile(r"[0-9a-f]{32}")
_SUFIJO = ".img"
_SUFIJO_PARCIAL = ".parcial"


class AlmacenTemporalEnDisco:
    def __init__(
        self,
        carpeta: Path,
        ttl_s: float,
        ahora: Callable[[], float] = time.time,
    ) -> None:
        self._carpeta = carpeta
        self._ttl_s = ttl_s
        self._ahora = ahora  # inyectable para probar el vencimiento sin esperar

    async def guardar(self, contenido: bytes) -> str:
        referencia = secrets.token_hex(16)
        await anyio.to_thread.run_sync(self._escribir, referencia, contenido)
        return referencia

    async def obtener(self, referencia: str) -> bytes | None:
        """`None` si no está o si ya venció su TTL. Una referencia que no tenga el formato propio se trata como
        inexistente: nunca se arma una ruta con texto de afuera (path traversal)."""
        if not _REFERENCIA.fullmatch(referencia):
            return None
        return await anyio.to_thread.run_sync(self._leer, referencia)

    async def borrar(self, referencia: str) -> None:
        """Idempotente: borrar lo que no está no es un error."""
        if _REFERENCIA.fullmatch(referencia):
            await anyio.to_thread.run_sync(self._ruta(referencia).unlink, True)

    async def barrer_vencidos(self) -> int:
        """Borra lo que pasó su TTL, incluidos restos de escrituras interrumpidas. Devuelve cuántos borró."""
        return await anyio.to_thread.run_sync(self._barrer)

    # --- internos (corren en un hilo) -----------------------------------------------------------

    def _ruta(self, referencia: str) -> Path:
        return self._carpeta / f"{referencia}{_SUFIJO}"

    def _vencio(self, ruta: Path) -> bool:
        return self._ahora() - ruta.stat().st_mtime > self._ttl_s

    def _escribir(self, referencia: str, contenido: bytes) -> None:
        self._carpeta.mkdir(parents=True, exist_ok=True)
        destino = self._ruta(referencia)
        parcial = destino.with_suffix(_SUFIJO_PARCIAL)
        parcial.write_bytes(contenido)
        parcial.replace(destino)  # atómico: nadie lee un archivo a medio escribir

    def _leer(self, referencia: str) -> bytes | None:
        ruta = self._ruta(referencia)
        try:
            if self._vencio(ruta):
                ruta.unlink(missing_ok=True)
                return None
            return ruta.read_bytes()
        except FileNotFoundError:
            return None

    def _barrer(self) -> int:
        if not self._carpeta.is_dir():
            return 0
        borrados = 0
        for ruta in self._carpeta.iterdir():
            if ruta.suffix not in (_SUFIJO, _SUFIJO_PARCIAL):
                continue  # no es nuestro: no se toca
            try:
                if self._vencio(ruta):
                    ruta.unlink(missing_ok=True)
                    borrados += 1
            except FileNotFoundError:
                continue  # lo borró otro hilo entre el listado y acá
        return borrados
