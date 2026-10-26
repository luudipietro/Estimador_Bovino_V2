"""Lectura de un upload por bloques con tope de tamaño. No depende de FastAPI."""

from collections.abc import Awaitable, Callable

from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio

TAMANO_BLOQUE = 1024 * 1024


async def leer_con_limite(
    leer: Callable[[int], Awaitable[bytes]],
    max_bytes: int,
    tamano_bloque: int = TAMANO_BLOQUE,
) -> bytes:
    """Lee hasta `max_bytes` y corta apenas se pasa, **antes** de seguir acumulando en memoria.

    `leer(n)` devuelve hasta `n` bytes y `b""` al terminar (es la firma de `UploadFile.read`). Se acepta
    exactamente `max_bytes`; con un byte más, `imagen_muy_grande`.
    """
    partes: list[bytes] = []
    total = 0
    while True:
        bloque = await leer(tamano_bloque)
        if not bloque:
            return b"".join(partes)
        total += len(bloque)
        if total > max_bytes:
            raise ErrorDeDominio(CodigoError.IMAGEN_MUY_GRANDE, "la imagen supera el tamaño máximo permitido")
        partes.append(bloque)
