from pathlib import Path

import pytest
from pydantic import ValidationError

from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio
from bv_estimacion.infraestructura.config import Configuracion
from bv_estimacion.infraestructura.imagenes.validacion import validar_imagen
from tests import imagenes_sinteticas as img
from tests.conftest import CARPETA_MODELOS_REPO, RAIZ_REPO

VARIABLES = [
    "TOKEN_SERVICIO", "SECRETO_HMAC", "HOSTS_IMAGEN_PERMITIDOS", "HOSTS_CALLBACK_PERMITIDOS", "CARPETA_MODELOS",
    "PRESUPUESTO_ESTIMACION_S", "HILOS_INFERENCIA", "MAX_MB_IMAGEN", "MAX_PIXELES", "RESOLUCION_MINIMA_MP",
    "CARPETA_TEMPORAL", "TTL_IMAGENES_S", "UMBRAL_BORROSIDAD", "TTL_TRABAJOS_S", "TAMANO_COLA",
    "INTENTOS_CALLBACK", "ESPERA_BASE_CALLBACK_S", "DOCS_HABILITADAS",
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
    assert cfg.resolucion_minima_mp == 0.75
    assert cfg.resolucion_minima_px == 750_000
    assert cfg.ttl_trabajos_s == 86_400
    assert cfg.tamano_cola == 200
    assert cfg.intentos_callback == 3
    assert cfg.espera_base_callback_s == 1.0
    assert cfg.hilos_inferencia >= 1
    assert cfg.hosts_imagen_permitidos == []
    assert cfg.hosts_callback_permitidos == []
    assert cfg.docs_habilitadas is False  # /docs no tiene autenticación


@pytest.mark.requisito("RNF-03")
@pytest.mark.parametrize(("nucleos", "esperado"), [(64, 4), (4, 4), (2, 2), (1, 1), (None, 1)])
def test_por_defecto_los_hilos_son_los_nucleos_con_tope_de_4(
    monkeypatch: pytest.MonkeyPatch, nucleos: int | None, esperado: int
) -> None:
    """Cada hilo carga su propia copia del modelo (~130 MB): sin tope, un servidor de 64 núcleos cargaría 8 GB."""
    monkeypatch.setattr("bv_estimacion.infraestructura.config.os.cpu_count", lambda: nucleos)
    assert configuracion(monkeypatch).hilos_inferencia == esperado


@pytest.mark.requisito("RNF-03")
def test_el_tope_de_hilos_es_solo_el_valor_por_defecto_y_se_puede_pedir_mas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bv_estimacion.infraestructura.config.os.cpu_count", lambda: 64)
    assert configuracion(monkeypatch, HILOS_INFERENCIA="8").hilos_inferencia == 8


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


@pytest.mark.requisito("RF-19")
def test_los_ttl_de_trabajos_e_imagenes_se_leen_del_entorno(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = configuracion(monkeypatch, TTL_TRABAJOS_S="7200", TTL_IMAGENES_S="900")
    assert cfg.ttl_trabajos_s == 7200
    assert cfg.ttl_imagenes_s == 900


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


RAIZ_SERVICIO = RAIZ_REPO / "servicio-estimacion"


@pytest.mark.requisito("RF-15")
def test_la_resolucion_minima_por_defecto_acepta_1024x768_y_rechaza_800x600(monkeypatch: pytest.MonkeyPatch) -> None:
    """Con el límite real de la configuración, no con un número suelto: 786 432 px pasan, 480 000 no."""
    cfg = configuracion(monkeypatch)
    limites = {
        "max_bytes": cfg.max_bytes_imagen,
        "max_pixeles": cfg.max_pixeles,
        "resolucion_minima_px": cfg.resolucion_minima_px,
    }

    validar_imagen(img.jpeg(1024, 768), **limites)  # no lanza

    with pytest.raises(ErrorDeDominio) as error:
        validar_imagen(img.jpeg(800, 600), **limites)
    assert error.value.codigo is CodigoError.RESOLUCION_INSUFICIENTE


@pytest.mark.requisito("RNF-02")
def test_carpeta_modelos_por_defecto_es_la_de_v1_del_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = configuracion(monkeypatch)
    assert cfg.carpeta_modelos == CARPETA_MODELOS_REPO
    assert cfg.carpeta_modelos.is_absolute()


@pytest.mark.requisito("RNF-02")
def test_carpeta_modelos_por_defecto_no_depende_del_directorio_de_trabajo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    assert configuracion(monkeypatch).carpeta_modelos == CARPETA_MODELOS_REPO


@pytest.mark.requisito("RNF-02")
def test_una_carpeta_modelos_relativa_se_resuelve_desde_el_proyecto_no_desde_el_directorio_de_trabajo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """La línea de v2 del `.env.example` (`../estimador/modelos_v2`) es relativa a `servicio-estimacion/`."""
    monkeypatch.chdir(tmp_path)
    cfg = configuracion(monkeypatch, CARPETA_MODELOS="../estimador/modelos_v2")
    assert cfg.carpeta_modelos == (RAIZ_SERVICIO / ".." / "estimador" / "modelos_v2").resolve()


@pytest.mark.requisito("RNF-02")
def test_una_carpeta_modelos_absoluta_se_respeta_tal_cual(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert configuracion(monkeypatch, CARPETA_MODELOS=str(tmp_path / "otra")).carpeta_modelos == tmp_path / "otra"
