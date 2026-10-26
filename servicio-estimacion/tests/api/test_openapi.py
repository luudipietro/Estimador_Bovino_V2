import json
import subprocess
import sys
from pathlib import Path

import pytest

from bv_estimacion.api.openapi import esquema_openapi, serializar_esquema
from bv_estimacion.dominio.errores import CodigoError
from tests.conftest import RAIZ_REPO

RAIZ_SERVICIO = RAIZ_REPO / "servicio-estimacion"
ARCHIVO = RAIZ_SERVICIO / "openapi.json"
SCRIPT = RAIZ_SERVICIO / "scripts" / "exportar_openapi.py"


@pytest.fixture(scope="module")
def esquema() -> dict[str, object]:
    return esquema_openapi()


def operaciones(esquema: dict[str, object]) -> list[tuple[str, str, dict[str, object]]]:
    rutas = esquema["paths"]
    assert isinstance(rutas, dict)
    return [(ruta, metodo, op) for ruta, ops in rutas.items() for metodo, op in ops.items()]


@pytest.mark.requisito("RF-22")
def test_el_openapi_versionado_es_el_que_genera_la_app() -> None:
    """Si esto falla cambió la API: regenerar con `python scripts/exportar_openapi.py` y avisar al equipo del
    adaptador C# (el archivo es lo que ellos leen)."""
    assert ARCHIVO.read_text(encoding="utf-8") == serializar_esquema(esquema_openapi())


@pytest.mark.requisito("RF-22")
def test_la_generacion_es_deterministica() -> None:
    assert serializar_esquema(esquema_openapi()) == serializar_esquema(esquema_openapi())


@pytest.mark.requisito("RF-22")
def test_el_archivo_es_json_utf8_que_termina_en_salto_de_linea() -> None:
    crudo = ARCHIVO.read_bytes()
    assert crudo.endswith(b"\n")
    assert json.loads(crudo.decode("utf-8"))["openapi"].startswith("3.")


@pytest.mark.requisito("RF-19")
def test_estan_todas_las_rutas_del_contrato(esquema: dict[str, object]) -> None:
    pares = {(ruta, metodo.upper()) for ruta, metodo, _ in operaciones(esquema)}
    assert pares == {
        ("/salud", "GET"),
        ("/v1/modelo", "GET"),
        ("/v1/estimaciones", "POST"),
        ("/v1/estimaciones/{id_trabajo}", "GET"),
        ("/v1/validaciones", "POST"),
    }


@pytest.mark.requisito("RF-22")
def test_toda_operacion_declara_el_bearer_salvo_salud(esquema: dict[str, object]) -> None:
    for ruta, metodo, operacion in operaciones(esquema):
        if ruta == "/salud":
            assert "security" not in operacion
        else:
            assert operacion["security"] == [{"BearerAuth": []}], f"{metodo} {ruta}"
    componentes = esquema["components"]
    assert isinstance(componentes, dict)
    assert componentes["securitySchemes"]["BearerAuth"] == {"type": "http", "scheme": "bearer"}


@pytest.mark.requisito("RF-22")
def test_los_errores_se_documentan_como_problem_json_con_los_codigos_del_catalogo(esquema: dict[str, object]) -> None:
    componentes = esquema["components"]
    assert isinstance(componentes, dict)
    problema = componentes["schemas"]["Problema"]
    assert set(problema["properties"]["codigo"]["enum"]) == {c.value for c in CodigoError}
    assert {"type", "title", "status", "detail", "codigo", "request_id"} <= set(problema["required"])

    for ruta, _, operacion in operaciones(esquema):
        if ruta == "/salud":
            continue
        for estado in ("401", "500"):
            contenido = operacion["responses"][estado]["content"]
            assert contenido["application/problem+json"]["schema"] == {"$ref": "#/components/schemas/Problema"}


@pytest.mark.requisito("RF-22")
def test_no_queda_el_error_422_de_fastapi_que_no_es_el_que_devuelve_el_servicio(esquema: dict[str, object]) -> None:
    """FastAPI documenta `422 HTTPValidationError`; este servicio devuelve `400 solicitud_invalida` en problem+json."""
    componentes = esquema["components"]
    assert isinstance(componentes, dict)
    assert "HTTPValidationError" not in componentes["schemas"]
    assert "ValidationError" not in componentes["schemas"]
    assert "HTTPValidationError" not in json.dumps(esquema)


@pytest.mark.requisito("RF-19")
def test_el_post_de_estimaciones_documenta_202_200_y_las_dos_formas_del_cuerpo(esquema: dict[str, object]) -> None:
    operacion = esquema["paths"]["/v1/estimaciones"]["post"]  # type: ignore[index]
    assert {"200", "202", "400", "409", "413", "415", "422", "503"} <= set(operacion["responses"])
    assert set(operacion["requestBody"]["content"]) == {"multipart/form-data", "application/json"}
    encabezados = [p["name"] for p in operacion["parameters"] if p["in"] == "header"]
    assert "Idempotency-Key" in encabezados


@pytest.mark.requisito("RF-19")
def test_el_get_de_un_trabajo_documenta_sus_tres_formas(esquema: dict[str, object]) -> None:
    operacion = esquema["paths"]["/v1/estimaciones/{id_trabajo}"]["get"]  # type: ignore[index]
    esquema_200 = operacion["responses"]["200"]["content"]["application/json"]["schema"]
    asignados = set(esquema_200["discriminator"]["mapping"])
    assert asignados == {"pendiente", "procesando", "completado", "fallido"}
    assert "404" in operacion["responses"]


# --- El script --------------------------------------------------------------------------------------


def correr(*argumentos: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-I", str(SCRIPT), *argumentos],
        capture_output=True,
        text=True,
        check=False,
        cwd=RAIZ_SERVICIO,
        timeout=120,
    )


@pytest.mark.requisito("RF-22")
def test_el_script_escribe_el_archivo_que_pide(tmp_path: Path) -> None:
    destino = tmp_path / "salida.json"
    resultado = correr("--salida", str(destino))

    assert resultado.returncode == 0, resultado.stderr
    assert destino.read_text(encoding="utf-8") == serializar_esquema(esquema_openapi())


@pytest.mark.requisito("RF-22")
def test_el_script_con_comprobar_dice_si_el_archivo_esta_al_dia(tmp_path: Path) -> None:
    destino = tmp_path / "salida.json"
    correr("--salida", str(destino))

    assert correr("--comprobar", "--salida", str(destino)).returncode == 0

    destino.write_text('{"desactualizado": true}\n', encoding="utf-8")
    desactualizado = correr("--comprobar", "--salida", str(destino))
    assert desactualizado.returncode == 1
    assert "scripts/exportar_openapi.py" in desactualizado.stderr


@pytest.mark.requisito("RF-22")
def test_el_script_con_comprobar_falla_si_el_archivo_no_existe(tmp_path: Path) -> None:
    assert correr("--comprobar", "--salida", str(tmp_path / "no-existe.json")).returncode == 1
