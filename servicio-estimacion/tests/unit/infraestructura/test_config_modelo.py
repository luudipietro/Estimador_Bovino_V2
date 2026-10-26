import json
from pathlib import Path

import pytest

from bv_estimacion.infraestructura.ml.config_modelo import (
    ConfigModeloInvalida,
    cargar_ficha_modelo,
    cargar_version_modelo,
)
from tests.conftest import CARPETA_MODELOS_REPO


def escribir(carpeta: Path, contenido: object) -> Path:
    (carpeta / "config.json").write_text(json.dumps(contenido), encoding="utf-8")
    return carpeta


@pytest.mark.requisito("RF-20")
def test_r20_r23_la_version_y_el_rango_salen_del_config_json(tmp_path: Path) -> None:
    carpeta = escribir(tmp_path, {"version": "2026-12-01 / datos argentinos", "rango_entrenamiento_kg": [80, 700]})
    version = cargar_version_modelo(carpeta)
    assert version.version == "2026-12-01 / datos argentinos"
    assert version.rango_validez_kg == (80.0, 700.0)


@pytest.mark.requisito("RF-20")
def test_el_config_json_versionado_del_repo_se_lee_sin_necesitar_los_pesos() -> None:
    esperado = json.loads((CARPETA_MODELOS_REPO / "config.json").read_text(encoding="utf-8"))
    version = cargar_version_modelo(CARPETA_MODELOS_REPO)
    assert version.version == esperado["version"]
    assert version.rango_validez_kg == tuple(float(x) for x in esperado["rango_entrenamiento_kg"])


@pytest.mark.requisito("RF-20")
@pytest.mark.parametrize(
    "contenido",
    [
        {"rango_entrenamiento_kg": [36, 621]},  # sin versión
        {"version": "v1"},  # sin rango
        {"version": "v1", "rango_entrenamiento_kg": [36]},  # rango incompleto
        {"version": "v1", "rango_entrenamiento_kg": [621, 36]},  # rango invertido
        {"version": "  ", "rango_entrenamiento_kg": [36, 621]},  # versión vacía
        {"version": "v1", "rango_entrenamiento_kg": "36-621"},  # tipo equivocado
    ],
)
def test_un_config_json_incompleto_o_invalido_es_un_error_claro(tmp_path: Path, contenido: object) -> None:
    with pytest.raises(ConfigModeloInvalida):
        cargar_version_modelo(escribir(tmp_path, contenido))


@pytest.mark.requisito("RF-20")
def test_sin_config_json_o_con_json_roto_es_un_error_claro(tmp_path: Path) -> None:
    with pytest.raises(ConfigModeloInvalida):
        cargar_version_modelo(tmp_path)
    (tmp_path / "config.json").write_text("{ no es json", encoding="utf-8")
    with pytest.raises(ConfigModeloInvalida):
        cargar_version_modelo(tmp_path)


BASE = {"version": "v-prueba", "rango_entrenamiento_kg": [36, 621]}


@pytest.mark.requisito("RF-20")
def test_la_ficha_trae_lo_opcional_cuando_el_config_json_lo_tiene(tmp_path: Path) -> None:
    carpeta = escribir(
        tmp_path,
        BASE | {"mape_test": 15.6, "ic95_test": [14.0, 16.3], "fecha_entrenamiento": "2026-09-25"},
    )
    ficha = cargar_ficha_modelo(carpeta)
    assert ficha.version.version == "v-prueba"
    assert ficha.mape_test == 15.6
    assert ficha.ic95_test == (14.0, 16.3)
    assert ficha.fecha_entrenamiento == "2026-09-25"


@pytest.mark.requisito("RF-20")
def test_la_ficha_deja_en_none_lo_que_el_config_json_no_tiene(tmp_path: Path) -> None:
    """El `config.json` de v1 no trae `fecha_entrenamiento`: no se inventa ni se saca de la versión (texto libre)."""
    ficha = cargar_ficha_modelo(escribir(tmp_path, BASE | {"version": "2026-09-25 / entrenado con otro dataset"}))
    assert ficha.mape_test is None
    assert ficha.ic95_test is None
    assert ficha.fecha_entrenamiento is None


@pytest.mark.requisito("RF-20")
@pytest.mark.parametrize(
    "extra",
    [
        {"mape_test": "mucho"},
        {"mape_test": True},
        {"ic95_test": [14.0]},
        {"ic95_test": "14-16"},
        {"fecha_entrenamiento": 20260925},
        {"fecha_entrenamiento": "  "},
    ],
)
def test_un_dato_opcional_con_el_tipo_equivocado_es_un_error_claro(tmp_path: Path, extra: dict[str, object]) -> None:
    with pytest.raises(ConfigModeloInvalida):
        cargar_ficha_modelo(escribir(tmp_path, BASE | extra))


@pytest.mark.requisito("RF-20")
def test_la_ficha_del_repo_se_lee_sin_necesitar_los_pesos() -> None:
    esperado = json.loads((CARPETA_MODELOS_REPO / "config.json").read_text(encoding="utf-8"))
    ficha = cargar_ficha_modelo(CARPETA_MODELOS_REPO)
    assert ficha.version == cargar_version_modelo(CARPETA_MODELOS_REPO)
    assert ficha.mape_test == esperado.get("mape_test")
