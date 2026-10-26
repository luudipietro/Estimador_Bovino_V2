"""Control de URLs de imágenes y de callback contra una allowlist (anti-SSRF). Lo usan el borde de la API y los
adaptadores HTTP, para que haya una sola regla."""

from collections.abc import Sequence
from urllib.parse import urlsplit

from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio


def validar_url_permitida(url: str, hosts_permitidos: Sequence[str]) -> None:
    """Solo `https`, sin credenciales en la URL y con el host exacto en la allowlist.

    Lanza `ErrorDeDominio(referencia_no_permitida)` sin repetir la URL en el mensaje: puede ser una URL firmada.

    Límite conocido: se valida el *nombre* del host, no la IP a la que resuelve. Los hosts de la lista tienen que ser
    infraestructura propia; si uno lo controla un tercero, un cambio de DNS puede apuntarlo a una red interna.
    """
    try:
        partes = urlsplit(url)
        host = (partes.hostname or "").lower()
        _ = partes.port  # un puerto inválido lanza ValueError al leerlo
    except ValueError as error:
        raise ErrorDeDominio(CodigoError.REFERENCIA_NO_PERMITIDA, "la URL no es válida") from error

    if partes.scheme != "https":
        raise ErrorDeDominio(CodigoError.REFERENCIA_NO_PERMITIDA, "solo se aceptan URLs https")
    if partes.username is not None or partes.password is not None:
        raise ErrorDeDominio(CodigoError.REFERENCIA_NO_PERMITIDA, "la URL no puede llevar credenciales")
    if not host or host not in {h.lower() for h in hosts_permitidos}:
        raise ErrorDeDominio(CodigoError.REFERENCIA_NO_PERMITIDA, "el host de la URL no está permitido")
