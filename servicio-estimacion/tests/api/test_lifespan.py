import os
import time
from pathlib import Path

import anyio
import httpx
import pytest
import respx
from fastapi import Depends

from bv_estimacion.api import main
from bv_estimacion.api.dependencias import Contenedor, obtener_contenedor
from bv_estimacion.api.main import create_app
from bv_estimacion.aplicacion.dto import SolicitarEstimacionCmd
from bv_estimacion.dominio.errores import MotivoNoSe
from bv_estimacion.dominio.estimacion import EstadoTrabajo
from bv_estimacion.dominio.valores import IdEvento, IdTrabajo
from bv_estimacion.infraestructura.ml.config_modelo import ConfigModeloInvalida
from bv_estimacion.infraestructura.ml.verificacion import VerificacionModeloFallida
from bv_estimacion.infraestructura.notificacion.callback_https import (
    ENCABEZADO_FIRMA,
    ENCABEZADO_TIMESTAMP,
    verificar_firma_callback,
)
from tests.modelo_falso import SECRETO, FabricaFalsa, configuracion_de_prueba


def contenedor_de(app: object) -> Contenedor:
    contenedor = app.state.contenedor  # type: ignore[attr-defined]
    assert isinstance(contenedor, Contenedor)
    return contenedor


async def esperar_estado(contenedor: Contenedor, id_trabajo: IdTrabajo, estado: EstadoTrabajo) -> None:
    with anyio.fail_after(5):
        while (await contenedor.consultar.ejecutar(id_trabajo)).estado is not estado:
            await anyio.sleep(0.01)


# --- Arranque -----------------------------------------------------------------------------------


@pytest.mark.requisito("RNF-03")
async def test_al_arrancar_carga_todas_las_instancias_antes_de_aceptar_trafico(tmp_path: Path) -> None:
    fabrica = FabricaFalsa()
    app = create_app(configuracion_de_prueba(tmp_path, hilos_inferencia=3), fabrica_modelo=fabrica.configurada())

    async with app.router.lifespan_context(app):
        contenedor = contenedor_de(app)
        assert contenedor.adaptador.cargado
        assert fabrica.creadas == 3


@pytest.mark.requisito("R-23")
async def test_la_version_y_el_rango_salen_del_config_json_de_la_carpeta_de_modelos(tmp_path: Path) -> None:
    app = create_app(configuracion_de_prueba(tmp_path), fabrica_modelo=FabricaFalsa().configurada())

    async with app.router.lifespan_context(app):
        ficha = contenedor_de(app).ficha
        assert ficha.version.version == "v-prueba / datos de prueba"
        assert ficha.version.rango_validez_kg == (36.0, 621.0)


@pytest.mark.requisito("RF-22")
async def test_con_un_peso_alterado_la_app_no_arranca_y_no_llega_a_cargar_el_modelo(tmp_path: Path) -> None:
    config = configuracion_de_prueba(tmp_path)
    (config.carpeta_modelos / "pose.pt").write_bytes(b"cambiado")
    fabrica = FabricaFalsa()
    app = create_app(config, fabrica_modelo=fabrica.configurada())

    with pytest.raises(VerificacionModeloFallida):
        async with app.router.lifespan_context(app):
            pytest.fail("no tendría que haber arrancado")

    assert fabrica.creadas == 0  # ni siquiera se intentó cargar


@pytest.mark.requisito("RF-22")
async def test_con_un_peso_faltante_la_app_no_arranca(tmp_path: Path) -> None:
    config = configuracion_de_prueba(tmp_path)
    (config.carpeta_modelos / "cnn.pt").unlink()
    app = create_app(config, fabrica_modelo=FabricaFalsa().configurada())

    with pytest.raises(VerificacionModeloFallida):
        async with app.router.lifespan_context(app):
            pytest.fail("no tendría que haber arrancado")


@pytest.mark.requisito("RF-22")
async def test_con_un_config_json_roto_la_app_no_arranca(tmp_path: Path) -> None:
    config = configuracion_de_prueba(tmp_path)
    (config.carpeta_modelos / "config.json").write_text("{ esto no es json", encoding="utf-8")
    app = create_app(config, fabrica_modelo=FabricaFalsa().configurada())

    with pytest.raises((VerificacionModeloFallida, ConfigModeloInvalida)):
        async with app.router.lifespan_context(app):
            pytest.fail("no tendría que haber arrancado")


@pytest.mark.requisito("RF-22")
async def test_si_el_modelo_no_carga_la_app_no_arranca_en_vez_de_aceptar_pedidos_que_van_a_fallar(
    tmp_path: Path,
) -> None:
    fabrica = FabricaFalsa()
    fabrica.error_al_crear = OSError("no se pudo abrir el archivo de pesos")
    app = create_app(configuracion_de_prueba(tmp_path), fabrica_modelo=fabrica.configurada())

    with pytest.raises(OSError, match="no se pudo abrir"):
        async with app.router.lifespan_context(app):
            pytest.fail("no tendría que haber arrancado")


@pytest.mark.requisito("RNF-03")
async def test_sin_estado_global_dos_apps_no_comparten_nada(tmp_path: Path) -> None:
    una = create_app(configuracion_de_prueba(tmp_path / "a"), fabrica_modelo=FabricaFalsa().configurada())
    otra = create_app(configuracion_de_prueba(tmp_path / "b"), fabrica_modelo=FabricaFalsa().configurada())

    async with una.router.lifespan_context(una), otra.router.lifespan_context(otra):
        assert contenedor_de(una) is not contenedor_de(otra)
        assert contenedor_de(una).solicitar is not contenedor_de(otra).solicitar


# --- Funcionamiento de punta a punta (sin HTTP) --------------------------------------------------


async def pedir(contenedor: Contenedor, id_evento: str, imagen: bytes, **cmd: object) -> IdTrabajo:
    aceptada = await contenedor.solicitar.ejecutar(
        SolicitarEstimacionCmd(id_evento=IdEvento(id_evento), imagenes=(imagen,), **cmd)  # type: ignore[arg-type]
    )
    return aceptada.trabajo.id


@pytest.mark.requisito("RF-19")
async def test_un_pedido_encolado_lo_procesa_un_worker_y_queda_completado(tmp_path: Path) -> None:
    fabrica = FabricaFalsa()
    app = create_app(configuracion_de_prueba(tmp_path), fabrica_modelo=fabrica.configurada())

    async with app.router.lifespan_context(app):
        contenedor = contenedor_de(app)
        id_trabajo = await pedir(contenedor, "evt-1", b"foto")

        await esperar_estado(contenedor, id_trabajo, EstadoTrabajo.COMPLETADO)

        trabajo = await contenedor.consultar.ejecutar(id_trabajo)
        assert trabajo.resultado is not None
        assert trabajo.resultado.peso_kg == 187.4
        assert trabajo.resultado.version == contenedor.ficha.version
        assert fabrica.llamadas == [b"foto"]


@pytest.mark.requisito("R-21")
async def test_un_no_se_del_modelo_llega_como_completado_no_confiable(tmp_path: Path) -> None:
    fabrica = FabricaFalsa()
    fabrica.respuesta_por_defecto = {
        "peso_kg": None,
        "confiable": False,
        "motivo": "no se detectó ningún bovino en la foto",
    }
    app = create_app(configuracion_de_prueba(tmp_path), fabrica_modelo=fabrica.configurada())

    async with app.router.lifespan_context(app):
        contenedor = contenedor_de(app)
        id_trabajo = await pedir(contenedor, "evt-1", b"campo")
        await esperar_estado(contenedor, id_trabajo, EstadoTrabajo.COMPLETADO)

        resultado = (await contenedor.consultar.ejecutar(id_trabajo)).resultado
        assert resultado is not None and not resultado.confiable
        assert resultado.motivos == (MotivoNoSe.SIN_ANIMAL,)


@pytest.mark.requisito("RF-22")
async def test_un_error_del_modelo_deja_el_trabajo_fallido_y_el_worker_sigue_vivo(tmp_path: Path) -> None:
    fabrica = FabricaFalsa()
    fabrica.error_al_estimar = RuntimeError("falló la inferencia")
    app = create_app(configuracion_de_prueba(tmp_path), fabrica_modelo=fabrica.configurada())

    async with app.router.lifespan_context(app):
        contenedor = contenedor_de(app)
        malo = await pedir(contenedor, "evt-1", b"uno")
        await esperar_estado(contenedor, malo, EstadoTrabajo.FALLIDO)

        fabrica.error_al_estimar = None  # el siguiente pedido anda: el worker no murió
        bueno = await pedir(contenedor, "evt-2", b"dos")
        await esperar_estado(contenedor, bueno, EstadoTrabajo.COMPLETADO)


@pytest.mark.requisito("RF-21")
@respx.mock
async def test_el_callback_sale_firmado_despues_de_guardar_el_resultado(tmp_path: Path) -> None:
    ruta = respx.post("https://backend.example/cb").mock(return_value=httpx.Response(200))
    app = create_app(configuracion_de_prueba(tmp_path), fabrica_modelo=FabricaFalsa().configurada())

    async with app.router.lifespan_context(app):
        contenedor = contenedor_de(app)
        id_trabajo = await pedir(contenedor, "evt-1", b"foto", url_notificacion="https://backend.example/cb")
        with anyio.fail_after(5):
            while not ruta.called:
                await anyio.sleep(0.01)

        pedido = ruta.calls.last.request
        assert verificar_firma_callback(
            SECRETO, pedido.headers[ENCABEZADO_TIMESTAMP], pedido.content, pedido.headers[ENCABEZADO_FIRMA]
        )
        assert b'"estado":"completado"' in pedido.content.replace(b" ", b"")
        assert str(id_trabajo).encode() in pedido.content


# --- Mantenimiento y apagado ---------------------------------------------------------------------


@pytest.mark.requisito("RF-19")
async def test_el_barrido_periodico_borra_las_imagenes_temporales_vencidas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "INTERVALO_BARRIDO_S", 0.02)
    config = configuracion_de_prueba(tmp_path, ttl_imagenes_s=1)
    app = create_app(config, fabrica_modelo=FabricaFalsa().configurada())

    async with app.router.lifespan_context(app):
        config.carpeta_temporal.mkdir(parents=True, exist_ok=True)
        vieja = config.carpeta_temporal / ("a" * 32 + ".img")
        vieja.write_bytes(b"resto de un proceso que se cayo")
        hace_un_dia = time.time() - 86_400
        os.utime(vieja, (hace_un_dia, hace_un_dia))

        with anyio.fail_after(5):
            while vieja.exists():
                await anyio.sleep(0.01)


@pytest.mark.requisito("RF-19")
async def test_al_apagar_se_cierra_el_cliente_http_y_se_detienen_las_tareas(tmp_path: Path) -> None:
    app = create_app(configuracion_de_prueba(tmp_path), fabrica_modelo=FabricaFalsa().configurada())

    with anyio.fail_after(5):  # si el apagado esperara a los workers para siempre, esto falla en vez de colgarse
        async with app.router.lifespan_context(app):
            cliente = contenedor_de(app).cliente
            assert not cliente.is_closed

    assert cliente.is_closed


@pytest.mark.requisito("RNF-03")
async def test_los_routers_obtienen_el_contenedor_con_depends(tmp_path: Path) -> None:
    app = create_app(configuracion_de_prueba(tmp_path), fabrica_modelo=FabricaFalsa().configurada())

    @app.get("/_version")
    async def version(contenedor: Contenedor = Depends(obtener_contenedor)) -> dict[str, str]:  # noqa: B008
        return {"version": contenedor.ficha.version.version}

    async with app.router.lifespan_context(app):
        transporte = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transporte, base_url="http://prueba") as cliente:
            respuesta = await cliente.get("/_version")

    assert respuesta.json() == {"version": "v-prueba / datos de prueba"}
