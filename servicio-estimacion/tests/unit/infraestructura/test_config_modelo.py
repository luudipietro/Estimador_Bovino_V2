import json
from pathlib import Path

import pytest

from bv_estimacion.infraestructura.ml.config_modelo import ConfigModeloInvalida, cargar_version_modelo
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
