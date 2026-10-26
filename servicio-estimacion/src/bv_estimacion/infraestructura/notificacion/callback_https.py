"""Callback HTTPS firmado al backend C# (implementa `EnviadorCallback`).

Contrato (ver `docs/contrato-backend.md`): `POST` al host de la allowlist con el mismo cuerpo que
`GET /v1/estimaciones/{id}`, firmado con HMAC-SHA256 sobre `<timestamp>.<cuerpo>`. El timestamp va **dentro** de lo
firmado: si solo se firmara el cuerpo, un callback capturado podría reenviarse con un timestamp nuevo y la firma
seguiría siendo válida, y el timestamp no protegería de ningún replay.
"""

import hashlib
import hmac
import logging
from collections.abc import Callable, Sequence

import httpx

from bv_estimacion.aplicacion.puertos import CallbackFallido, Reloj
from bv_estimacion.dominio.errores import ErrorDeDominio
from bv_estimacion.dominio.estimacion import Trabajo
from bv_estimacion.infraestructura.urls_permitidas import validar_url_permitida

logger = logging.getLogger(__name__)

ENCABEZADO_FIRMA = "X-BV-Firma"
ENCABEZADO_TIMESTAMP = "X-BV-Timestamp"
_PREFIJO = "sha256="


def firmar_callback(secreto: str, timestamp: str, cuerpo: bytes) -> str:
    """Devuelve el valor del encabezado `X-BV-Firma`: `sha256=<hex del HMAC de "<timestamp>.<cuerpo>">`."""
    mensaje = timestamp.encode("ascii") + b"." + cuerpo
    return _PREFIJO + hmac.new(secreto.encode("utf-8"), mensaje, hashlib.sha256).hexdigest()


def verificar_firma_callback(secreto: str, timestamp: str, cuerpo: bytes, firma_recibida: str) -> bool:
    """Compara en tiempo constante. Solo verifica la firma: rechazar timestamps viejos (anti-replay) es decisión de
    quien recibe, que conoce la hora y la ventana que tolera."""
    return hmac.compare_digest(firmar_callback(secreto, timestamp, cuerpo), firma_recibida)


class EnviadorCallbackHttp:
    """Un intento de entrega por llamada. Los reintentos con backoff los hace `NotificarResultado`.

    El cuerpo lo arma `serializar`, que en la Fase 4 será la misma función que arma la respuesta del `GET`, para que
    el callback y la consulta digan siempre lo mismo. El cliente lo crea y lo cierra quien arma la app.
    """

    def __init__(
        self,
        cliente: httpx.AsyncClient,
        hosts_permitidos: Sequence[str],
        secreto: str,
        serializar: Callable[[Trabajo], bytes],
        reloj: Reloj,
    ) -> None:
        self._cliente = cliente
        self._hosts = list(hosts_permitidos)
        self._secreto = secreto
        self._serializar = serializar
        self._reloj = reloj

    async def enviar(self, url: str, trabajo: Trabajo) -> None:
        try:
            validar_url_permitida(url, self._hosts)
        except ErrorDeDominio as error:  # la allowlist pudo cambiar entre el POST y el aviso
            raise CallbackFallido("la URL del callback no está permitida") from error

        cuerpo = self._serializar(trabajo)
        timestamp = str(int(self._reloj.ahora().timestamp()))  # segundos Unix; se recalcula en cada intento
        encabezados = {
            "Content-Type": "application/json",
            ENCABEZADO_TIMESTAMP: timestamp,
            ENCABEZADO_FIRMA: firmar_callback(self._secreto, timestamp, cuerpo),
        }
        try:
            respuesta = await self._cliente.post(url, content=cuerpo, headers=encabezados, follow_redirects=False)
        except httpx.HTTPError as error:
            # Solo la clase del error: ni la URL (puede llevar datos del backend) ni el detalle.
            logger.warning("callback_http_error", extra={"error": type(error).__name__})
            raise CallbackFallido("no se pudo contactar al backend") from error
        if not respuesta.is_success:
            raise CallbackFallido(f"el backend respondió {respuesta.status_code}")
