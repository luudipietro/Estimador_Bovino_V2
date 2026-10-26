import io
import threading
from typing import Any

import anyio
import numpy as np
import pytest
from PIL import Image

from bv_estimacion.aplicacion.puertos import ModeloNoDisponible
from bv_estimacion.dominio.errores import MotivoNoSe
from bv_estimacion.dominio.valores import VersionModelo
from bv_estimacion.infraestructura.ml.adaptador_modelo import AdaptadorModelo
from bv_estimacion.infraestructura.ml.motivos import MotivoDesconocido
from tests.imagenes_sinteticas import jpeg

VERSION = VersionModelo("v-prueba", 36.0, 621.0)
LADO_MAX = 1024

TEXTO_CORTADO = "el animal aparece cortado por el borde de la foto"
TEXTO_FUERA_DE_RANGO = "el peso estimado cae fuera del rango con el que se entrenó el modelo"


def respuesta(peso: float = 187.4, **cambios: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "peso_kg": peso,
        "confiable": True,
        "motivo": None,
        "confianza_deteccion": 0.94,
        "confianza_keypoint_minima": 0.88,
        "bovinos_en_la_escena": 1,
        "recorte": None,
        "resolucion": "100x100",
        "modelo": "v-prueba",
    }
    return base | cambios


class EstimadorFalso:
    """Reemplaza a `EstimadorPeso`: no importa torch ni ultralytics. Responde según los bytes que recibe."""

    def __init__(self, respuestas: dict[bytes, dict[str, Any]], numero: int) -> None:
        self.respuestas = respuestas
        self.numero = numero
        self.llamadas: list[bytes] = []
        self.frenar: threading.Event | None = None
        self.arranco = threading.Event()

    def estimar(self, imagen: bytes) -> dict[str, Any]:
        self.llamadas.append(imagen)
        self.arranco.set()
        if self.frenar is not None:
            self.frenar.wait(5)
        return self.respuestas[imagen]


class Fabrica:
    def __init__(self) -> None:
        self.respuestas: dict[bytes, dict[str, Any]] = {}
        self.creados: list[EstimadorFalso] = []
        self.hilos: list[int] = []
        self.error: Exception | None = None

    def __call__(self) -> EstimadorFalso:
        if self.error is not None:
            raise self.error
        self.hilos.append(threading.get_ident())
        estimador = EstimadorFalso(self.respuestas, len(self.creados) + 1)
        self.creados.append(estimador)
        return estimador


def png_con_borde(nitida: bool) -> bytes:
    """Una foto 'con animal' nítida (tablero) o lisa (sin detalle)."""
    if nitida:
        celdas = (np.indices((200, 200)).sum(axis=0) // 4 % 2 * 255).astype(np.uint8)
        imagen = Image.fromarray(celdas).convert("RGB")
    else:
        imagen = Image.new("RGB", (200, 200), (120, 90, 60))
    buffer = io.BytesIO()
    imagen.save(buffer, "PNG")
    return buffer.getvalue()


async def adaptador(
    fabrica: Fabrica | None = None, *, cantidad: int = 1, umbral_borrosidad: float | None = None
) -> tuple[AdaptadorModelo, Fabrica]:
    fabrica = fabrica or Fabrica()
    caso = AdaptadorModelo(fabrica, cantidad, VERSION, lado_max_px=LADO_MAX, umbral_borrosidad=umbral_borrosidad)
    await caso.cargar()
    return caso, fabrica


# --- Carga --------------------------------------------------------------------------------------


@pytest.mark.requisito("RNF-03")
async def test_no_esta_cargado_hasta_que_cargan_todas_las_instancias() -> None:
    fabrica = Fabrica()
    caso = AdaptadorModelo(fabrica, 3, VERSION, lado_max_px=LADO_MAX, umbral_borrosidad=None)
    assert not caso.cargado
    assert fabrica.creados == []  # construir el adaptador no carga nada

    await caso.cargar()

    assert caso.cargado
    assert len(fabrica.creados) == 3


@pytest.mark.requisito("RNF-03")
async def test_informa_cuantas_instancias_tiene_cargadas() -> None:
    fabrica = Fabrica()
    caso = AdaptadorModelo(fabrica, 3, VERSION, lado_max_px=LADO_MAX, umbral_borrosidad=None)
    assert caso.instancias == 0
    await caso.cargar()
    assert caso.instancias == 3


@pytest.mark.requisito("RNF-03")
async def test_la_carga_corre_fuera_del_event_loop() -> None:
    _, fabrica = await adaptador(cantidad=2)
    assert threading.get_ident() not in fabrica.hilos


@pytest.mark.requisito("RNF-03")
async def test_si_una_instancia_no_carga_el_adaptador_sigue_sin_estar_cargado() -> None:
    fabrica = Fabrica()
    fabrica.error = OSError("falta el archivo de pesos")
    caso = AdaptadorModelo(fabrica, 2, VERSION, lado_max_px=LADO_MAX, umbral_borrosidad=None)

    with pytest.raises(OSError, match="falta el archivo"):
        await caso.cargar()

    assert not caso.cargado


@pytest.mark.requisito("RNF-03")
async def test_estimar_antes_de_cargar_es_modelo_no_disponible() -> None:
    caso = AdaptadorModelo(Fabrica(), 1, VERSION, lado_max_px=LADO_MAX, umbral_borrosidad=None)
    with pytest.raises(ModeloNoDisponible):
        await caso.estimar(b"x")


@pytest.mark.requisito("RNF-03")
async def test_cargar_dos_veces_no_duplica_instancias() -> None:
    caso, fabrica = await adaptador(cantidad=2)
    await caso.cargar()
    assert len(fabrica.creados) == 2


@pytest.mark.requisito("R-23")
async def test_expone_la_version_del_modelo() -> None:
    caso, _ = await adaptador()
    assert caso.version == VERSION


@pytest.mark.parametrize("cantidad", [0, -1])
def test_hace_falta_al_menos_una_instancia(cantidad: int) -> None:
    with pytest.raises(ValueError, match="al menos una instancia"):
        AdaptadorModelo(Fabrica(), cantidad, VERSION, lado_max_px=LADO_MAX, umbral_borrosidad=None)


# --- Traducción del resultado -------------------------------------------------------------------


@pytest.mark.requisito("RF-20")
async def test_una_foto_confiable_se_traduce_con_sus_confianzas() -> None:
    caso, fabrica = await adaptador()
    fabrica.respuestas[b"foto"] = respuesta(187.4)

    foto = (await caso.estimar(b"foto")).resultado

    assert foto.peso_kg == 187.4
    assert foto.confiable
    assert foto.motivos == ()
    assert foto.confianza_deteccion == 0.94
    assert foto.confianza_keypoint_minima == 0.88
    assert foto.bovinos_en_la_escena == 1


@pytest.mark.requisito("R-21")
async def test_sin_animal_es_no_se_con_ese_motivo_y_no_un_error() -> None:
    caso, fabrica = await adaptador()
    fabrica.respuestas[b"campo"] = {
        "peso_kg": None,
        "confiable": False,
        "motivo": "no se detectó ningún bovino en la foto",
        "bovinos_en_la_escena": 0,
        "resolucion": "100x100",
    }

    foto = (await caso.estimar(b"campo")).resultado

    assert foto.peso_kg is None
    assert not foto.confiable
    assert foto.motivos == (MotivoNoSe.SIN_ANIMAL,)
    assert foto.bovinos_en_la_escena == 0


@pytest.mark.requisito("R-21")
async def test_un_motivo_del_modulo_se_traduce_y_conserva_el_peso() -> None:
    caso, fabrica = await adaptador()
    fabrica.respuestas[b"foto"] = respuesta(150.0, confiable=False, motivo=TEXTO_CORTADO)

    foto = (await caso.estimar(b"foto")).resultado

    assert foto.motivos == (MotivoNoSe.ANIMAL_CORTADO,)
    assert not foto.confiable
    assert foto.peso_kg == 150.0


@pytest.mark.requisito("RF-22")
async def test_un_motivo_desconocido_falla_en_voz_alta() -> None:
    caso, fabrica = await adaptador()
    fabrica.respuestas[b"foto"] = respuesta(150.0, confiable=False, motivo="motivo nuevo del modulo")
    with pytest.raises(MotivoDesconocido):
        await caso.estimar(b"foto")


# --- R-20: el rango sale de config.json, no del módulo -------------------------------------------


@pytest.mark.requisito("R-20")
async def test_un_peso_fuera_del_rango_de_config_no_es_confiable_aunque_el_modulo_no_lo_diga() -> None:
    caso, fabrica = await adaptador()
    fabrica.respuestas[b"foto"] = respuesta(700.0)  # el módulo lo da por confiable

    foto = (await caso.estimar(b"foto")).resultado

    assert not foto.confiable
    assert foto.motivos == (MotivoNoSe.FUERA_DE_RANGO,)
    assert foto.peso_kg == 700.0


@pytest.mark.requisito("R-20")
async def test_el_motivo_fuera_de_rango_del_modulo_cede_ante_el_rango_de_config() -> None:
    """Si `estimador.py` y `config.json` discrepan en el rango, manda `config.json` (es la fuente de verdad)."""
    caso, fabrica = await adaptador()
    fabrica.respuestas[b"foto"] = respuesta(150.0, confiable=False, motivo=TEXTO_FUERA_DE_RANGO)

    foto = (await caso.estimar(b"foto")).resultado

    assert foto.confiable
    assert foto.motivos == ()


@pytest.mark.requisito("R-20")
async def test_no_duplica_el_motivo_fuera_de_rango() -> None:
    caso, fabrica = await adaptador()
    fabrica.respuestas[b"foto"] = respuesta(700.0, confiable=False, motivo=TEXTO_FUERA_DE_RANGO)
    assert (await caso.estimar(b"foto")).resultado.motivos == (MotivoNoSe.FUERA_DE_RANGO,)


# --- R-24: borrosidad ---------------------------------------------------------------------------


@pytest.mark.requisito("R-24")
async def test_sin_umbral_la_regla_no_se_aplica_ni_se_decodifica_la_imagen() -> None:
    caso, fabrica = await adaptador(umbral_borrosidad=None)
    fabrica.respuestas[b"no-es-una-imagen"] = respuesta(150.0, recorte=[0, 0, 50, 50])

    foto = (await caso.estimar(b"no-es-una-imagen")).resultado

    assert foto.confiable  # si hubiera intentado medir, habría fallado al decodificar


@pytest.mark.requisito("R-24")
async def test_una_foto_lisa_con_umbral_es_borrosa_y_no_confiable() -> None:
    caso, fabrica = await adaptador(umbral_borrosidad=100.0)
    lisa = png_con_borde(nitida=False)
    fabrica.respuestas[lisa] = respuesta(150.0, recorte=[0, 0, 200, 200])

    foto = (await caso.estimar(lisa)).resultado

    assert not foto.confiable
    assert foto.motivos == (MotivoNoSe.IMAGEN_BORROSA,)
    assert foto.peso_kg == 150.0


@pytest.mark.requisito("R-24")
async def test_una_foto_nitida_con_umbral_sigue_siendo_confiable() -> None:
    caso, fabrica = await adaptador(umbral_borrosidad=100.0)
    nitida = png_con_borde(nitida=True)
    fabrica.respuestas[nitida] = respuesta(150.0, recorte=[0, 0, 200, 200])

    assert (await caso.estimar(nitida)).resultado.confiable


@pytest.mark.requisito("R-24")
async def test_la_borrosidad_se_suma_a_los_motivos_del_modulo() -> None:
    caso, fabrica = await adaptador(umbral_borrosidad=100.0)
    lisa = png_con_borde(nitida=False)
    fabrica.respuestas[lisa] = respuesta(150.0, confiable=False, motivo=TEXTO_CORTADO)

    motivos = (await caso.estimar(lisa)).resultado.motivos

    assert motivos == (MotivoNoSe.ANIMAL_CORTADO, MotivoNoSe.IMAGEN_BORROSA)


@pytest.mark.requisito("R-24")
async def test_sin_animal_no_se_mide_borrosidad() -> None:
    caso, fabrica = await adaptador(umbral_borrosidad=100.0)
    fabrica.respuestas[b"campo-vacio"] = {
        "peso_kg": None,
        "confiable": False,
        "motivo": "no se detectó ningún bovino en la foto",
    }
    assert (await caso.estimar(b"campo-vacio")).resultado.motivos == (MotivoNoSe.SIN_ANIMAL,)


# --- Advertencias -------------------------------------------------------------------------------


@pytest.mark.requisito("RF-22")
async def test_una_foto_sin_exif_trae_la_advertencia() -> None:
    caso, fabrica = await adaptador()
    foto = jpeg()
    fabrica.respuestas[foto] = respuesta()
    assert (await caso.estimar(foto)).advertencias == ("no trae metadata EXIF",)


@pytest.mark.requisito("RF-22")
async def test_una_foto_con_exif_no_trae_advertencias() -> None:
    caso, fabrica = await adaptador()
    foto = jpeg(con_exif=True)
    fabrica.respuestas[foto] = respuesta()
    assert (await caso.estimar(foto)).advertencias == ()


@pytest.mark.requisito("RF-22")
async def test_si_no_se_puede_leer_el_exif_no_se_inventa_una_advertencia() -> None:
    caso, fabrica = await adaptador()
    fabrica.respuestas[b"no-es-una-imagen"] = respuesta()
    assert (await caso.estimar(b"no-es-una-imagen")).advertencias == ()


# --- Concurrencia -------------------------------------------------------------------------------


@pytest.mark.requisito("RNF-03")
async def test_con_dos_instancias_dos_fotos_se_estiman_a_la_vez_en_instancias_distintas() -> None:
    caso, fabrica = await adaptador(cantidad=2)
    fabrica.respuestas[b"a"] = respuesta(100.0)
    fabrica.respuestas[b"b"] = respuesta(200.0)
    soltar = threading.Event()
    for estimador in fabrica.creados:
        estimador.frenar = soltar

    pesos: list[float | None] = []

    async def estimar(contenido: bytes) -> None:
        pesos.append((await caso.estimar(contenido)).resultado.peso_kg)

    async with anyio.create_task_group() as grupo:
        grupo.start_soon(estimar, b"a")
        grupo.start_soon(estimar, b"b")
        with anyio.fail_after(3):  # las dos tienen que estar adentro antes de soltar a ninguna
            while not all(e.arranco.is_set() for e in fabrica.creados):
                await anyio.sleep(0.005)
        soltar.set()

    assert sorted(p for p in pesos if p is not None) == [100.0, 200.0]
    assert [len(e.llamadas) for e in fabrica.creados] == [1, 1]


@pytest.mark.requisito("RNF-01")
async def test_un_timeout_no_libera_la_instancia_hasta_que_el_hilo_termina() -> None:
    caso, fabrica = await adaptador(cantidad=1)
    fabrica.respuestas[b"lenta"] = respuesta()
    fabrica.respuestas[b"siguiente"] = respuesta(99.0)
    estimador = fabrica.creados[0]
    estimador.frenar = threading.Event()

    with anyio.move_on_after(0.05) as alcance:  # presupuesto de tiempo del caso de uso
        await caso.estimar(b"lenta")
    assert alcance.cancelled_caught

    with anyio.move_on_after(0.1) as espera:  # el siguiente espera el lugar: no arranca otro hilo
        await caso.estimar(b"siguiente")
    assert espera.cancelled_caught
    assert estimador.llamadas == [b"lenta"]

    estimador.frenar.set()
    with anyio.fail_after(3):
        assert (await caso.estimar(b"siguiente")).resultado.peso_kg == 99.0
