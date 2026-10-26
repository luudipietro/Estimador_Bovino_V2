import pytest

from bv_estimacion.aplicacion.casos_uso.validar_imagen import ValidarImagen
from bv_estimacion.aplicacion.dto import EstimacionFoto
from bv_estimacion.dominio.errores import MotivoNoSe
from bv_estimacion.dominio.estimacion import ResultadoFoto
from tests.fakes import ModeloFalso, foto_ok


def no_se(*motivos: MotivoNoSe, peso: float | None = None) -> EstimacionFoto:
    return EstimacionFoto(ResultadoFoto(peso_kg=peso, confiable=False, motivos=motivos))


@pytest.mark.requisito("RF-15")
async def test_una_foto_apta_se_valida() -> None:
    resultado = await ValidarImagen(ModeloFalso()).ejecutar(b"foto")
    assert resultado.apta
    assert resultado.motivos == ()


@pytest.mark.requisito("RF-15")
async def test_los_motivos_de_calidad_de_la_imagen_se_informan() -> None:
    modelo = ModeloFalso()
    modelo.por_defecto = no_se(MotivoNoSe.IMAGEN_BORROSA, MotivoNoSe.ANIMAL_CORTADO)

    resultado = await ValidarImagen(modelo).ejecutar(b"foto")

    assert not resultado.apta
    assert resultado.motivos == (MotivoNoSe.IMAGEN_BORROSA, MotivoNoSe.ANIMAL_CORTADO)


@pytest.mark.requisito("RF-15")
async def test_sin_animal_no_es_apta() -> None:
    modelo = ModeloFalso()
    modelo.por_defecto = no_se(MotivoNoSe.SIN_ANIMAL)
    assert not (await ValidarImagen(modelo).ejecutar(b"foto")).apta


@pytest.mark.requisito("RF-15")
async def test_fuera_de_rango_no_es_un_problema_de_la_imagen_y_no_se_informa() -> None:
    """R-20 depende del peso estimado y pertenece a la estimación, no a la validación de la imagen."""
    modelo = ModeloFalso()
    modelo.por_defecto = no_se(MotivoNoSe.FUERA_DE_RANGO, peso=680.0)

    resultado = await ValidarImagen(modelo).ejecutar(b"foto")

    assert resultado.apta
    assert resultado.motivos == ()


@pytest.mark.requisito("RF-15")
async def test_fuera_de_rango_junto_a_un_problema_real_solo_informa_el_problema() -> None:
    modelo = ModeloFalso()
    modelo.por_defecto = no_se(MotivoNoSe.FUERA_DE_RANGO, MotivoNoSe.IMAGEN_BORROSA, peso=680.0)

    resultado = await ValidarImagen(modelo).ejecutar(b"foto")

    assert not resultado.apta
    assert resultado.motivos == (MotivoNoSe.IMAGEN_BORROSA,)


@pytest.mark.requisito("RF-15")
async def test_las_advertencias_de_la_imagen_pasan_a_la_validacion() -> None:
    modelo = ModeloFalso()
    modelo.por_defecto = foto_ok(187.4, "no trae metadata EXIF")
    resultado = await ValidarImagen(modelo).ejecutar(b"foto")
    assert resultado.advertencias == ("no trae metadata EXIF",)
