import pytest
from PIL import Image

from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio
from bv_estimacion.infraestructura.imagenes.validacion import (
    FORMATO_HEIC,
    FORMATO_JPEG,
    FORMATO_PNG,
    detectar_formato,
    validar_imagen,
)
from tests import imagenes_sinteticas as img

LIMITES = {"max_bytes": 5 * 1024 * 1024, "max_pixeles": 1_000_000, "resolucion_minima_px": 1}


def codigo_de(datos: bytes, **cambios: int) -> CodigoError:
    with pytest.raises(ErrorDeDominio) as error:
        validar_imagen(datos, **{**LIMITES, **cambios})
    return error.value.codigo


# --- Formato (415) ------------------------------------------------------------------------------


@pytest.mark.requisito("RF-15")
def test_se_detecta_el_formato_por_los_magic_bytes() -> None:
    assert detectar_formato(img.jpeg()) == FORMATO_JPEG
    assert detectar_formato(img.png()) == FORMATO_PNG


@pytest.mark.requisito("RF-15")
@pytest.mark.parametrize("datos", [img.gif(), img.bmp(), b"%PDF-1.7 no soy una imagen", b"\x00" * 64])
def test_gif_bmp_y_otros_formatos_son_formato_no_admitido(datos: bytes) -> None:
    assert detectar_formato(datos) is None
    assert codigo_de(datos) is CodigoError.FORMATO_NO_ADMITIDO


@pytest.mark.requisito("RF-15")
def test_el_formato_lo_dicen_los_bytes_y_no_la_extension() -> None:
    """Un GIF con nombre `.jpg` sigue siendo un GIF: la función ni recibe el nombre."""
    assert codigo_de(img.gif()) is CodigoError.FORMATO_NO_ADMITIDO


@pytest.mark.requisito("RF-15")
def test_un_jpeg_y_un_png_validos_se_aceptan_y_devuelven_sus_dimensiones() -> None:
    resultado = validar_imagen(img.jpeg(120, 80, con_exif=True), **LIMITES)
    assert (resultado.formato, resultado.ancho, resultado.alto) == (FORMATO_JPEG, 120, 80)
    assert validar_imagen(img.png(50, 40), **LIMITES).formato == FORMATO_PNG


@pytest.mark.requisito("RF-15")
def test_un_heic_valido_se_acepta_r08() -> None:
    try:
        datos = img.heic()
    except Exception:  # noqa: BLE001 - si esta instalación de libheif no puede codificar, no hay qué probar
        pytest.skip("esta instalación de pillow-heif no puede generar un HEIC de prueba")
    assert detectar_formato(datos) == FORMATO_HEIC
    resultado = validar_imagen(datos, **LIMITES)
    assert (resultado.formato, resultado.ancho, resultado.alto) == (FORMATO_HEIC, 100, 100)


# --- Corrupción (422) ---------------------------------------------------------------------------


@pytest.mark.requisito("RF-22")
def test_un_archivo_vacio_es_imagen_corrupta() -> None:
    assert codigo_de(b"") is CodigoError.IMAGEN_CORRUPTA


@pytest.mark.requisito("RF-22")
def test_un_jpeg_truncado_es_imagen_corrupta() -> None:
    completo = img.jpeg(300, 300, ruido=True)
    assert codigo_de(completo[: len(completo) // 2]) is CodigoError.IMAGEN_CORRUPTA


@pytest.mark.requisito("RF-22")
def test_bytes_con_cabecera_de_jpeg_pero_sin_imagen_son_imagen_corrupta() -> None:
    assert codigo_de(b"\xff\xd8\xff" + b"basura" * 50) is CodigoError.IMAGEN_CORRUPTA


@pytest.mark.requisito("RF-22")
def test_un_png_con_la_cabecera_pero_dañado_es_imagen_corrupta() -> None:
    assert codigo_de(img.png()[:30]) is CodigoError.IMAGEN_CORRUPTA


@pytest.mark.requisito("RF-22")
def test_los_mensajes_de_error_no_filtran_detalles_internos() -> None:
    with pytest.raises(ErrorDeDominio) as error:
        validar_imagen(b"\xff\xd8\xff" + b"basura" * 50, **LIMITES)
    mensaje = str(error.value)
    assert "Traceback" not in mensaje
    assert "PIL" not in mensaje
    assert "Error" not in mensaje


# --- Tamaño y píxeles (413) ---------------------------------------------------------------------


@pytest.mark.requisito("RF-22")
def test_una_imagen_que_supera_el_limite_de_bytes_es_imagen_muy_grande() -> None:
    datos = img.jpeg(200, 200, ruido=True)
    assert codigo_de(datos, max_bytes=len(datos) - 1) is CodigoError.IMAGEN_MUY_GRANDE


@pytest.mark.requisito("RF-22")
def test_el_limite_de_bytes_se_chequea_antes_de_mirar_el_formato() -> None:
    assert codigo_de(b"\x00" * 100, max_bytes=10) is CodigoError.IMAGEN_MUY_GRANDE


@pytest.mark.requisito("RF-22")
def test_una_imagen_justo_en_el_limite_de_bytes_se_acepta() -> None:
    datos = img.jpeg()
    validar_imagen(datos, **{**LIMITES, "max_bytes": len(datos)})


@pytest.mark.requisito("RF-22")
def test_demasiados_pixeles_segun_el_limite_configurado_es_imagen_muy_grande() -> None:
    assert codigo_de(img.jpeg(100, 100), max_pixeles=9_999) is CodigoError.IMAGEN_MUY_GRANDE


@pytest.mark.requisito("RF-22")
def test_justo_en_el_limite_de_pixeles_se_acepta() -> None:
    validar_imagen(img.jpeg(100, 100), **{**LIMITES, "max_pixeles": 10_000})


@pytest.mark.requisito("RF-22")
def test_una_bomba_de_descompresion_de_pillow_se_traduce_a_imagen_muy_grande(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pillow corta por su cuenta si la imagen duplica su propio límite: eso también es `imagen_muy_grande`."""
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    datos = img.png(50, 50)  # 2.500 px > 2 * 100
    assert codigo_de(datos, max_pixeles=10**12) is CodigoError.IMAGEN_MUY_GRANDE


# --- Resolución (422) ---------------------------------------------------------------------------


@pytest.mark.requisito("RF-15")
def test_una_resolucion_por_debajo_del_minimo_es_resolucion_insuficiente() -> None:
    assert codigo_de(img.jpeg(100, 100), resolucion_minima_px=10_001) is CodigoError.RESOLUCION_INSUFICIENTE


@pytest.mark.requisito("RF-15")
def test_justo_en_la_resolucion_minima_se_acepta() -> None:
    validar_imagen(img.jpeg(100, 100), **{**LIMITES, "resolucion_minima_px": 10_000})


@pytest.mark.requisito("RF-15")
def test_la_corrupcion_se_detecta_antes_que_la_resolucion() -> None:
    completo = img.jpeg(300, 300, ruido=True)
    assert codigo_de(completo[: len(completo) // 2], resolucion_minima_px=10**9) is CodigoError.IMAGEN_CORRUPTA


# --- EXIF (R-07) --------------------------------------------------------------------------------


@pytest.mark.requisito("RF-15")
def test_r07_una_foto_sin_exif_genera_advertencia() -> None:
    resultado = validar_imagen(img.jpeg(con_exif=False), **LIMITES)
    assert resultado.advertencias == ("no trae metadata EXIF",)


@pytest.mark.requisito("RF-15")
def test_r07_una_foto_con_exif_no_genera_advertencia() -> None:
    assert validar_imagen(img.jpeg(con_exif=True), **LIMITES).advertencias == ()


@pytest.mark.requisito("RF-15")
def test_un_png_tampoco_trae_exif_y_lo_advierte() -> None:
    assert validar_imagen(img.png(), **LIMITES).advertencias == ("no trae metadata EXIF",)
