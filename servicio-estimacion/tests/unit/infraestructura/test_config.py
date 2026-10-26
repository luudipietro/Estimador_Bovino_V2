from pathlib import Path

import pytest
from pydantic import ValidationError

from bv_estimacion.infraestructura.config import Configuracion

VARIABLES = [
    "TOKEN_SERVICIO", "SECRETO_HMAC", "HOSTS_IMAGEN_PERMITIDOS", "HOSTS_CALLBACK_PERMITIDOS", "CARPETA_MODELOS",
    "PRESUPUESTO_ESTIMACION_S", "HILOS_INFERENCIA", "MAX_MB_IMAGEN", "MAX_PIXELES", "RESOLUCION_MINIMA_MP",
    "CARPETA_TEMPORAL", "TTL_IMAGENES_S", "UMBRAL_BORROSIDAD", "TTL_TRABAJOS_S", "TAMANO_COLA",
    "INTENTOS_CALLBACK", "ESPERA_BASE_CALLBACK_S",
]  # fmt: skip


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Que el entorno de la máquina que corre los tests no cambie el resultado."""
    for variable in VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def configuracion(monkeypatch: pytest.MonkeyPatch, **entorno: str) -> Configuracion:
    base = {"TOKEN_SERVICIO": "token-de-prueba", "SECRETO_HMAC": "secreto-de-prueba", **entorno}
    for nombre, valor in base.items():
        monkeypatch.setenv(nombre, valor)
    return Configuracion(_env_file=None)  # type: ignore[call-arg]  # _env_file es un argumento propio de pydantic-settings


@pytest.mark.requisito("RNF-01")
def test_los_valores_por_defecto_son_los_documentados(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = configuracion(monkeypatch)
    assert cfg.presupuesto_estimacion_s == 45.0
    assert cfg.max_mb_imagen == 25
    assert cfg.resolucion_minima_mp == 3.0
    assert cfg.ttl_trabajos_s == 86_400
    assert cfg.tamano_cola == 200
    assert cfg.intentos_callback == 3
    assert cfg.espera_base_callback_s == 1.0
    assert cfg.hilos_inferencia >= 1
    assert cfg.hosts_imagen_permitidos == []
    assert cfg.hosts_callback_permitidos == []


@pytest.mark.requisito("RF-21")
def test_el_token_y_el_secreto_son_obligatorios(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        Configuracion(_env_file=None)  # type: ignore[call-arg]
    monkeypatch.setenv("TOKEN_SERVICIO", "t")
    with pytest.raises(ValidationError):  # falta el secreto
        Configuracion(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.requisito("RF-21")
def test_el_token_o_el_secreto_vacios_no_se_aceptan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRETO_HMAC", "s")
    monkeypatch.setenv("TOKEN_SERVICIO", "")
    with pytest.raises(ValidationError):
        Configuracion(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.requisito("RF-19")
def test_las_allowlists_se_leen_separadas_por_coma_sin_espacios_ni_mayusculas(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = configuracion(
        monkeypatch,
        HOSTS_IMAGEN_PERMITIDOS=" Almacen.Example , otro.example ,,",
        HOSTS_CALLBACK_PERMITIDOS="backend.example",
    )
    assert cfg.hosts_imagen_permitidos == ["almacen.example", "otro.example"]
    assert cfg.hosts_callback_permitidos == ["backend.example"]


@pytest.mark.requisito("RF-19")
def test_una_allowlist_vacia_no_acepta_ningun_host(monkeypatch: pytest.MonkeyPatch) -> None:
    assert configuracion(monkeypatch, HOSTS_IMAGEN_PERMITIDOS="").hosts_imagen_permitidos == []


@pytest.mark.requisito("RF-15")
def test_r24_sin_umbral_de_borrosidad_queda_en_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert configuracion(monkeypatch).umbral_borrosidad is None


@pytest.mark.requisito("RF-15")
def test_r24_umbral_vacio_en_el_entorno_tambien_es_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert configuracion(monkeypatch, UMBRAL_BORROSIDAD="").umbral_borrosidad is None


@pytest.mark.requisito("RF-15")
def test_r24_el_umbral_se_lee_cuando_esta_configurado(monkeypatch: pytest.MonkeyPatch) -> None:
    assert configuracion(monkeypatch, UMBRAL_BORROSIDAD="85.5").umbral_borrosidad == 85.5


@pytest.mark.requisito("RF-21")
def test_el_callback_se_configura_por_entorno(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = configuracion(monkeypatch, INTENTOS_CALLBACK="5", ESPERA_BASE_CALLBACK_S="0.5")
    assert (cfg.intentos_callback, cfg.espera_base_callback_s) == (5, 0.5)


@pytest.mark.requisito("RF-21")
@pytest.mark.parametrize("variable", ["INTENTOS_CALLBACK", "TAMANO_COLA", "HILOS_INFERENCIA", "MAX_MB_IMAGEN"])
def test_los_enteros_no_aceptan_cero(monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
    with pytest.raises(ValidationError):
        configuracion(monkeypatch, **{variable: "0"})


@pytest.mark.requisito("RNF-01")
def test_el_presupuesto_tiene_que_ser_positivo(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        configuracion(monkeypatch, PRESUPUESTO_ESTIMACION_S="0")


@pytest.mark.requisito("RF-15")
def test_los_limites_de_imagen_se_derivan_de_la_configuracion(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = configuracion(monkeypatch, MAX_MB_IMAGEN="2", RESOLUCION_MINIMA_MP="0.5", MAX_PIXELES="1000")
    assert cfg.max_bytes_imagen == 2 * 1024 * 1024
    assert cfg.resolucion_minima_px == 500_000
    assert cfg.max_pixeles == 1000


@pytest.mark.requisito("RF-19")
def test_las_carpetas_se_configuran_como_rutas(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = configuracion(monkeypatch, CARPETA_TEMPORAL=str(tmp_path / "tmp"), CARPETA_MODELOS=str(tmp_path / "m"))
    assert cfg.carpeta_temporal == tmp_path / "tmp"
    assert cfg.carpeta_modelos == tmp_path / "m"
