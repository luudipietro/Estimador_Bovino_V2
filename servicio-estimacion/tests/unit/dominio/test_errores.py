import pytest

from bv_estimacion.dominio.errores import (
    CODIGOS_PROPUESTOS,
    FALLAS_INTERNAS,
    CodigoError,
    ErrorDeDominio,
    MotivoNoSe,
    TransicionInvalida,
)

# Códigos de docs/errores.md. Si el catálogo cambia, este test obliga a cambiar el contrato a propósito.
CODIGOS_CONFIRMADOS = {
    "formato_no_admitido",
    "imagen_corrupta",
    "imagen_muy_grande",
    "resolucion_insuficiente",
    "cantidad_imagenes_invalida",
    "id_evento_en_conflicto",
    "referencia_no_permitida",
    "no_autenticado",
    "modelo_no_disponible",
    "tiempo_excedido",
    "referencia_inaccesible",
    "error_interno",
}
CODIGOS_NO_CONFIRMADOS = {"trabajo_no_encontrado", "cola_llena", "solicitud_invalida"}
MOTIVOS = {"sin_animal", "deteccion_dudosa", "partes_tapadas", "animal_cortado", "imagen_borrosa", "fuera_de_rango"}


@pytest.mark.requisito("RF-22")
def test_el_catalogo_confirmado_coincide_con_docs_errores() -> None:
    confirmados = {c.value for c in CodigoError} - {c.value for c in CODIGOS_PROPUESTOS}
    assert confirmados == CODIGOS_CONFIRMADOS


@pytest.mark.requisito("RF-22")
def test_los_codigos_propuestos_se_listan_aparte() -> None:
    """Si el equipo rechaza alguno, se borra del enum y de CODIGOS_PROPUESTOS (cambio de una línea)."""
    assert {c.value for c in CODIGOS_PROPUESTOS} == CODIGOS_NO_CONFIRMADOS


@pytest.mark.requisito("RF-22")
def test_el_catalogo_de_motivos_del_no_se_es_cerrado() -> None:
    assert {m.value for m in MotivoNoSe} == MOTIVOS


@pytest.mark.requisito("RF-22")
def test_los_codigos_son_snake_case() -> None:
    for codigo in [*CodigoError, *MotivoNoSe]:
        assert codigo.value == codigo.value.lower()
        assert " " not in codigo.value


@pytest.mark.requisito("RF-22")
def test_solo_cuatro_codigos_dejan_un_trabajo_fallido() -> None:
    assert {c.value for c in FALLAS_INTERNAS} == {
        "modelo_no_disponible",
        "tiempo_excedido",
        "referencia_inaccesible",
        "error_interno",
    }


@pytest.mark.requisito("RF-22")
def test_error_de_dominio_lleva_codigo_y_regla() -> None:
    error = ErrorDeDominio(CodigoError.CANTIDAD_IMAGENES_INVALIDA, "llegaron 5", regla="R-18")
    assert error.codigo is CodigoError.CANTIDAD_IMAGENES_INVALIDA
    assert error.regla == "R-18"
    assert str(error) == "llegaron 5"


@pytest.mark.requisito("RF-22")
def test_transicion_invalida_es_un_error_interno() -> None:
    assert TransicionInvalida("x").codigo is CodigoError.ERROR_INTERNO
