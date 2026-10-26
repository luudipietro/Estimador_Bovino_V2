import pytest

from bv_estimacion.dominio.calidad import (
    MedicionesFoto,
    UmbralesCalidad,
    animal_cortado,
    es_borrosa,
    evaluar_calidad,
)
from bv_estimacion.dominio.errores import MotivoNoSe

UMBRALES = UmbralesCalidad(confianza_deteccion_min=0.35, confianza_keypoint_min=0.50)


def medicion(**cambios: object) -> MedicionesFoto:
    """Una foto apta: animal completo, bien detectado, en el centro."""
    base: dict[str, object] = {
        "animal_detectado": True,
        "confianza_deteccion": 0.94,
        "confianza_keypoint_minima": 0.88,
        "caja_animal": (300.0, 200.0, 900.0, 700.0),
        "ancho": 1200,
        "alto": 900,
    }
    return MedicionesFoto(**{**base, **cambios})  # type: ignore[arg-type]  # el dict heterogéneo es de test


# --- R-24 ---------------------------------------------------------------------------------------


@pytest.mark.requisito("RF-15")
def test_r24_varianza_por_debajo_del_umbral_es_borrosa() -> None:
    assert es_borrosa(varianza_laplaciano=40.0, umbral=100.0)


@pytest.mark.requisito("RF-15")
def test_r24_varianza_igual_o_mayor_al_umbral_no_es_borrosa() -> None:
    assert not es_borrosa(100.0, 100.0)
    assert not es_borrosa(250.0, 100.0)


@pytest.mark.requisito("RF-15")
def test_r24_sin_umbral_la_regla_no_se_aplica() -> None:
    """Umbral None = todavía no calibrado: ninguna foto se marca borrosa, ni la más borrosa."""
    assert not es_borrosa(varianza_laplaciano=0.0, umbral=None)


@pytest.mark.requisito("RF-15")
def test_r24_sin_medicion_no_se_puede_decidir() -> None:
    assert not es_borrosa(varianza_laplaciano=None, umbral=100.0)


@pytest.mark.requisito("RF-15")
def test_r24_sin_umbral_evaluar_calidad_no_devuelve_imagen_borrosa() -> None:
    motivos = evaluar_calidad(medicion(varianza_laplaciano=0.0), UMBRALES)
    assert motivos == ()
    assert MotivoNoSe.IMAGEN_BORROSA.value == "imagen_borrosa"  # el código sigue en el catálogo


@pytest.mark.requisito("RF-15")
def test_r24_con_umbral_evaluar_calidad_marca_imagen_borrosa() -> None:
    umbrales = UmbralesCalidad(0.35, 0.50, umbral_borrosidad=100.0)
    assert evaluar_calidad(medicion(varianza_laplaciano=40.0), umbrales) == (MotivoNoSe.IMAGEN_BORROSA,)


# --- R-25 ---------------------------------------------------------------------------------------


@pytest.mark.requisito("RF-15")
def test_r25_una_foto_apta_no_tiene_motivos() -> None:
    assert evaluar_calidad(medicion(), UMBRALES) == ()


@pytest.mark.requisito("RF-15")
def test_r25_sin_animal_es_el_unico_motivo() -> None:
    mediciones = medicion(animal_detectado=False, confianza_deteccion=0.0, confianza_keypoint_minima=0.0)
    assert evaluar_calidad(mediciones, UMBRALES) == (MotivoNoSe.SIN_ANIMAL,)


@pytest.mark.requisito("RF-15")
def test_r25_deteccion_por_debajo_del_minimo_es_dudosa() -> None:
    assert evaluar_calidad(medicion(confianza_deteccion=0.34), UMBRALES) == (MotivoNoSe.DETECCION_DUDOSA,)
    assert evaluar_calidad(medicion(confianza_deteccion=0.35), UMBRALES) == ()


@pytest.mark.requisito("RF-15")
def test_r25_keypoint_por_debajo_del_minimo_es_parte_tapada() -> None:
    assert evaluar_calidad(medicion(confianza_keypoint_minima=0.49), UMBRALES) == (MotivoNoSe.PARTES_TAPADAS,)
    assert evaluar_calidad(medicion(confianza_keypoint_minima=0.50), UMBRALES) == ()


@pytest.mark.requisito("RF-15")
@pytest.mark.parametrize(
    "caja",
    [
        (5.0, 200.0, 900.0, 700.0),  # izquierda
        (300.0, 5.0, 900.0, 700.0),  # arriba
        (300.0, 200.0, 1195.0, 700.0),  # derecha
        (300.0, 200.0, 900.0, 895.0),  # abajo
    ],
)
def test_r25_animal_que_toca_el_borde_esta_cortado(caja: tuple[float, float, float, float]) -> None:
    assert animal_cortado(caja, ancho=1200, alto=900)


@pytest.mark.requisito("RF-15")
def test_r25_animal_descentrado_pero_completo_es_valido() -> None:
    assert not animal_cortado((200.0, 100.0, 600.0, 500.0), ancho=1200, alto=900)  # a un costado, sin tocar el borde
    assert evaluar_calidad(medicion(caja_animal=(200.0, 100.0, 600.0, 500.0)), UMBRALES) == ()


@pytest.mark.requisito("RF-15")
def test_r25_sin_caja_no_se_puede_decir_que_este_cortado() -> None:
    assert not animal_cortado(None, ancho=1200, alto=900)


@pytest.mark.requisito("RF-15")
def test_r25_varios_motivos_se_acumulan_en_orden_estable() -> None:
    mediciones = medicion(
        confianza_deteccion=0.2,
        confianza_keypoint_minima=0.3,
        caja_animal=(0.0, 200.0, 900.0, 700.0),
    )
    assert evaluar_calidad(mediciones, UMBRALES) == (
        MotivoNoSe.DETECCION_DUDOSA,
        MotivoNoSe.PARTES_TAPADAS,
        MotivoNoSe.ANIMAL_CORTADO,
    )
