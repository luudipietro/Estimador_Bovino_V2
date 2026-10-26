"""El adaptador con los pesos reales. Se saltea si los `.pt` no están (`python ../estimador/descargar_modelos.py`).

Solo comprueba que las piezas encajan con `EstimadorPeso` de verdad: carga, hilos y forma del resultado. No mide
la precisión del modelo (eso es `scripts/evaluar_modelo.py`).
"""

from pathlib import Path

import anyio
import pytest

from bv_estimacion.dominio.errores import MotivoNoSe
from bv_estimacion.infraestructura.ml.adaptador_modelo import AdaptadorModelo
from bv_estimacion.infraestructura.ml.config_modelo import cargar_version_modelo
from tests.imagenes_sinteticas import jpeg

pytestmark = pytest.mark.modelo


async def cargar_adaptador(carpeta: Path, cantidad: int) -> AdaptadorModelo:
    from estimador import LADO_MAX, EstimadorPeso  # recién acá: arrastra torch y ultralytics

    adaptador = AdaptadorModelo(
        lambda: EstimadorPeso(carpeta, dispositivo="cpu"),
        cantidad,
        cargar_version_modelo(carpeta),
        lado_max_px=LADO_MAX,
        umbral_borrosidad=None,
    )
    await adaptador.cargar()
    return adaptador


@pytest.mark.requisito("RNF-02")
async def test_carga_y_estima_una_foto_sin_animal_como_no_se(carpeta_modelos_reales: Path) -> None:
    adaptador = await cargar_adaptador(carpeta_modelos_reales, 1)
    assert adaptador.cargado

    foto = (await adaptador.estimar(jpeg(640, 480))).resultado  # un color liso: no hay ningún bovino

    assert not foto.confiable
    assert foto.motivos  # el "no sé" siempre dice por qué (R-21)
    assert all(isinstance(m, MotivoNoSe) for m in foto.motivos)


@pytest.mark.requisito("RNF-03")
async def test_dos_fotos_a_la_vez_con_dos_instancias_terminan_sin_pisarse(carpeta_modelos_reales: Path) -> None:
    adaptador = await cargar_adaptador(carpeta_modelos_reales, 2)
    resultados: list[object] = []

    async def estimar() -> None:
        resultados.append((await adaptador.estimar(jpeg(640, 480))).resultado)

    with anyio.fail_after(120):
        async with anyio.create_task_group() as grupo:
            grupo.start_soon(estimar)
            grupo.start_soon(estimar)

    assert len(resultados) == 2
