"""Descarga de imágenes desde URLs firmadas del almacenamiento privado del backend (implementa `ObtenedorImagen`)."""

import logging
from collections.abc import Sequence

import httpx

from bv_estimacion.aplicacion.puertos import ImagenNoDisponible
from bv_estimacion.dominio.errores import ErrorDeDominio
from bv_estimacion.infraestructura.urls_permitidas import validar_url_permitida

logger = logging.getLogger(__name__)


class ObtenedorImagenHttp:
    """Nunca sigue redirecciones (un host permitido podría redirigir a uno interno), corta al pasar `max_bytes` y
    traduce cualquier fallo a `ImagenNoDisponible`. Ni los mensajes ni los logs llevan la URL: puede ir firmada.

    El cliente lo crea y lo cierra quien arma la app (el `lifespan`), con los timeouts configurados.
    """

    def __init__(self, cliente: httpx.AsyncClient, hosts_permitidos: Sequence[str], max_bytes: int) -> None:
        self._cliente = cliente
        self._hosts = list(hosts_permitidos)
        self._max_bytes = max_bytes

    async def obtener(self, url: str) -> bytes:
        try:
            validar_url_permitida(url, self._hosts)
        except ErrorDeDominio as error:  # la allowlist pudo cambiar entre el POST y el worker
            raise ImagenNoDisponible("la URL ya no está permitida") from error

        try:
            async with self._cliente.stream("GET", url, follow_redirects=False) as respuesta:
                if respuesta.status_code != httpx.codes.OK:
                    raise ImagenNoDisponible(f"el almacenamiento respondió {respuesta.status_code}")
                declarado = respuesta.headers.get("content-length")
                if declarado is not None and declarado.isdigit() and int(declarado) > self._max_bytes:
                    raise ImagenNoDisponible("la imagen supera el tamaño máximo permitido")

                partes: list[bytes] = []
                total = 0
                async for bloque in respuesta.aiter_bytes():
                    total += len(bloque)
                    if total > self._max_bytes:
                        raise ImagenNoDisponible("la imagen supera el tamaño máximo permitido")
                    partes.append(bloque)
                return b"".join(partes)
        except ImagenNoDisponible:
            raise
        except httpx.HTTPError as error:
            # Timeout, conexión rechazada, etc. Se registra solo la clase, sin la URL ni el detalle.
            logger.warning("descarga_fallida", extra={"error": type(error).__name__})
            raise ImagenNoDisponible("no se pudo descargar la imagen") from error
