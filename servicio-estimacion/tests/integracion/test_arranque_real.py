"""La app completa con los pesos reales: cuánto tarda en arrancar y que un pedido real llegue de punta a punta.

Se saltea si los `.pt` no están (`python ../estimador/descargar_modelos.py`). No mide precisión (eso es
`scripts/evaluar_modelo.py`): usa una foto lisa, donde el modelo tiene que decir "no sé".
"""

import time
from pathlib import Path

import httpx
import pytest

from bv_estimacion.api.main import create_app
from tests.api.ayudas import AUTORIZACION, RUTA, encabezados, esperar_estado, foto
from tests.modelo_falso import configuracion_de_prueba

pytestmark = pytest.mark.modelo


@pytest.mark.requisito("RNF-01")
@pytest.mark.parametrize("instancias", [1, 4])
async def test_la_app_arranca_con_el_modelo_real_y_atiende_un_pedido(
    carpeta_modelos_reales: Path, tmp_path: Path, instancias: int, capsys: pytest.CaptureFixture[str]
) -> None:
    config = configuracion_de_prueba(tmp_path, carpeta_modelos=carpeta_modelos_reales, hilos_inferencia=instancias)
    app = create_app(config, configurar_logs=False)  # sin fábrica falsa: carga `EstimadorPeso` de verdad

    inicio = time.perf_counter()
    async with app.router.lifespan_context(app):
        arranque_s = time.perf_counter() - inicio
        transporte = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transporte, base_url="http://prueba") as cliente:
            salud = await cliente.get("/salud")
            assert salud.status_code == 200
            assert salud.json()["instancias"] == instancias

            modelo = (await cliente.get("/v1/modelo", headers=AUTORIZACION)).json()
            assert modelo["version"]
            assert modelo["rango_validez_kg"][0] < modelo["rango_validez_kg"][1]

            aceptado = await cliente.post(
                RUTA, headers=encabezados(), files=[("imagenes", ("lisa.jpg", foto(), "image/jpeg"))]
            )
            assert aceptado.status_code == 202
            inicio_pedido = time.perf_counter()
            final = (await esperar_estado(cliente, aceptado.json()["id"])).json()
            pedido_s = time.perf_counter() - inicio_pedido

    assert final["estado"] == "completado"  # un "no sé" también es completado (R-21)
    assert final["confiable"] is False
    assert final["motivos"]  # y siempre dice por qué

    with capsys.disabled():  # para leerlo con `pytest -m modelo -s`
        print(f"\nARRANQUE instancias={instancias}: {arranque_s:.1f} s | primer pedido: {pedido_s:.1f} s")
