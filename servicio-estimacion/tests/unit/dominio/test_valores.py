import dataclasses

import pytest

from bv_estimacion.dominio.valores import IdEvento, IdTrabajo, OrigenImagen, ReferenciaImagen, VersionModelo


@pytest.mark.requisito("RF-19")
@pytest.mark.parametrize("clase", [IdEvento, IdTrabajo])
def test_los_ids_no_aceptan_valores_vacios(clase: type[IdEvento] | type[IdTrabajo]) -> None:
    with pytest.raises(ValueError):
        clase("   ")


@pytest.mark.requisito("RF-19")
def test_los_ids_son_inmutables_y_comparables_por_valor() -> None:
    id_evento = IdEvento("c3b1")
    assert id_evento == IdEvento("c3b1")
    assert str(id_evento) == "c3b1"
    assert str(IdTrabajo("trb_01")) == "trb_01"
    with pytest.raises(dataclasses.FrozenInstanceError):
        id_evento.valor = "otro"  # type: ignore[misc]  # se prueba justamente que no se pueda asignar


@pytest.mark.requisito("RF-20")
def test_r20_los_extremos_del_rango_cuentan_como_validos() -> None:
    version = VersionModelo("v1", 36.0, 621.0)
    assert version.en_rango(36.0)
    assert version.en_rango(621.0)
    assert not version.en_rango(35.9)
    assert not version.en_rango(621.1)


@pytest.mark.requisito("RF-20")
def test_r20_el_rango_de_validez_se_expone_como_par() -> None:
    assert VersionModelo("v1", 36.0, 621.0).rango_validez_kg == (36.0, 621.0)


@pytest.mark.requisito("RF-20")
def test_r23_la_version_rechaza_rangos_y_nombres_invalidos() -> None:
    with pytest.raises(ValueError):
        VersionModelo("v1", 100.0, 100.0)
    with pytest.raises(ValueError):
        VersionModelo("  ", 36.0, 621.0)


# --- ReferenciaImagen ---------------------------------------------------------------------------


@pytest.mark.requisito("RF-19")
def test_la_referencia_acepta_un_sha256_de_64_hexadecimales_y_lo_normaliza_a_minusculas() -> None:
    referencia = ReferenciaImagen(OrigenImagen.URL, "https://almacen.example/img.jpg", "AB" * 32)
    assert referencia.sha256 == "ab" * 32


@pytest.mark.requisito("RF-19")
@pytest.mark.parametrize(
    "sha256",
    ["", "abc", "a" * 63, "a" * 65, "g" * 64, "z" * 64, " " + "a" * 63],
)
def test_la_referencia_rechaza_un_sha256_mal_formado(sha256: str) -> None:
    with pytest.raises(ValueError):
        ReferenciaImagen(OrigenImagen.URL, "https://almacen.example/img.jpg", sha256)


@pytest.mark.requisito("RF-19")
def test_la_referencia_rechaza_una_ubicacion_vacia() -> None:
    with pytest.raises(ValueError):
        ReferenciaImagen(OrigenImagen.ALMACEN_TEMPORAL, "  ", "a" * 64)


@pytest.mark.requisito("RF-19")
def test_la_referencia_es_inmutable() -> None:
    referencia = ReferenciaImagen(OrigenImagen.ALMACEN_TEMPORAL, "tmp-1", "a" * 64)
    with pytest.raises(dataclasses.FrozenInstanceError):
        referencia.sha256 = "b" * 64  # type: ignore[misc]  # se prueba justamente que no se pueda asignar
