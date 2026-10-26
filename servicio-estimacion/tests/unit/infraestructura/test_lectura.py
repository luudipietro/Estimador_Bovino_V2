import io

import pytest

from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio
from bv_estimacion.infraestructura.imagenes.lectura import leer_con_limite


class Fuente:
    """Imita `UploadFile.read(n)` y cuenta cuántas veces se la leyó."""

    def __init__(self, contenido: bytes) -> None:
        self._buffer = io.BytesIO(contenido)
        self.lecturas = 0

    async def leer(self, n: int) -> bytes:
        self.lecturas += 1
        return self._buffer.read(n)


@pytest.mark.requisito("RF-22")
async def test_lee_el_contenido_completo_en_bloques() -> None:
    contenido = bytes(range(256)) * 100
    fuente = Fuente(contenido)
    assert await leer_con_limite(fuente.leer, max_bytes=30_000, tamano_bloque=1000) == contenido
    assert fuente.lecturas == 27  # 25.600 bytes en bloques de 1.000 son 26 lecturas con datos, más la vacía final


@pytest.mark.requisito("RF-22")
async def test_exactamente_el_maximo_se_acepta() -> None:
    contenido = b"x" * 5000
    assert await leer_con_limite(Fuente(contenido).leer, max_bytes=5000, tamano_bloque=1024) == contenido


@pytest.mark.requisito("RF-22")
async def test_un_byte_de_mas_es_imagen_muy_grande() -> None:
    with pytest.raises(ErrorDeDominio) as error:
        await leer_con_limite(Fuente(b"x" * 5001).leer, max_bytes=5000, tamano_bloque=1024)
    assert error.value.codigo is CodigoError.IMAGEN_MUY_GRANDE


@pytest.mark.requisito("RF-22")
async def test_corta_apenas_se_pasa_y_no_sigue_leyendo_un_archivo_enorme() -> None:
    fuente = Fuente(b"x" * 1_000_000)
    with pytest.raises(ErrorDeDominio):
        await leer_con_limite(fuente.leer, max_bytes=2048, tamano_bloque=1024)
    assert fuente.lecturas == 3  # 1.024 + 1.024 + el bloque que se pasa; el resto no se leyó


@pytest.mark.requisito("RF-22")
async def test_un_archivo_vacio_devuelve_bytes_vacios_y_lo_decide_la_validacion() -> None:
    assert await leer_con_limite(Fuente(b"").leer, max_bytes=100) == b""
