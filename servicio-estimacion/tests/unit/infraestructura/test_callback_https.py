import hashlib
import hmac
import json
import logging

import httpx
import pytest
import respx

from bv_estimacion.aplicacion.casos_uso.notificar_resultado import NotificarResultado
from bv_estimacion.aplicacion.puertos import CallbackFallido
from bv_estimacion.dominio.estimacion import Trabajo
from bv_estimacion.dominio.valores import IdEvento, IdTrabajo, OrigenImagen, ReferenciaImagen
from bv_estimacion.infraestructura.notificacion.callback_https import (
    ENCABEZADO_FIRMA,
    ENCABEZADO_TIMESTAMP,
    EnviadorCallbackHttp,
    firmar_callback,
    verificar_firma_callback,
)
from tests.fakes import T0, DormidorFalso, RelojFijo

URL = "https://backend.example/api/estimaciones/callback?token=DATO-SENSIBLE"
RUTA = "https://backend.example/api/estimaciones/callback"
SECRETO = "secreto-compartido-de-prueba"
TIMESTAMP_T0 = str(int(T0.timestamp()))


def trabajo() -> Trabajo:
    referencia = ReferenciaImagen(OrigenImagen.ALMACEN_TEMPORAL, "tmp-1", "a" * 64)
    return Trabajo.crear(IdTrabajo("trb_01"), IdEvento("evt-1"), [referencia], T0, url_notificacion=URL)


def serializar(t: Trabajo) -> bytes:
    return json.dumps({"id": str(t.id), "id_evento": str(t.id_evento), "estado": t.estado.value}).encode()


def enviador(hosts: list[str] | None = None, reloj: RelojFijo | None = None) -> tuple[EnviadorCallbackHttp, RelojFijo]:
    reloj = reloj or RelojFijo()
    cliente = httpx.AsyncClient(timeout=5.0)
    hosts = ["backend.example"] if hosts is None else hosts
    return EnviadorCallbackHttp(cliente, hosts, SECRETO, serializar, reloj), reloj


def firma_independiente(secreto: str, timestamp: str, cuerpo: bytes) -> str:
    """Segunda implementación escrita aparte: el test no valida la función contra sí misma."""
    mensaje = f"{timestamp}.".encode() + cuerpo
    return "sha256=" + hmac.new(secreto.encode(), mensaje, hashlib.sha256).hexdigest()


# --- La firma -----------------------------------------------------------------------------------


@pytest.mark.requisito("RF-21")
def test_la_firma_es_hmac_sha256_de_timestamp_punto_cuerpo() -> None:
    cuerpo = b'{"estado": "completado"}'
    assert firmar_callback(SECRETO, "1793016000", cuerpo) == firma_independiente(SECRETO, "1793016000", cuerpo)
    assert firmar_callback(SECRETO, "1793016000", cuerpo).startswith("sha256=")


@pytest.mark.requisito("RF-21")
def test_el_timestamp_esta_firmado_y_cambiarlo_invalida_la_firma() -> None:
    """Sin esto un callback capturado se reenvía con un timestamp nuevo y la firma sigue valiendo (replay)."""
    cuerpo = b"{}"
    firma = firmar_callback(SECRETO, "1793016000", cuerpo)
    assert verificar_firma_callback(SECRETO, "1793016000", cuerpo, firma)
    assert not verificar_firma_callback(SECRETO, "1793016001", cuerpo, firma)


@pytest.mark.requisito("RF-21")
def test_la_firma_no_se_confunde_moviendo_el_limite_entre_timestamp_y_cuerpo() -> None:
    assert firmar_callback(SECRETO, "1793016000", b"5.x") != firmar_callback(SECRETO, "17930160005", b"x")


@pytest.mark.requisito("RF-21")
@pytest.mark.parametrize(
    ("secreto", "cuerpo", "firma_valida"),
    [
        (SECRETO, b"{}", True),
        ("otro-secreto", b"{}", False),  # secreto distinto
        (SECRETO, b'{"x": 1}', False),  # cuerpo alterado
    ],
)
def test_verificar_firma_acepta_solo_la_correcta(secreto: str, cuerpo: bytes, firma_valida: bool) -> None:
    firma = firmar_callback(SECRETO, "1793016000", b"{}")
    assert verificar_firma_callback(secreto, "1793016000", cuerpo, firma) is firma_valida


@pytest.mark.requisito("RF-21")
@pytest.mark.parametrize("firma", ["", "sha256=", "deadbeef", "md5=abc", "SHA256=" + "0" * 64])
def test_verificar_firma_rechaza_formatos_mal_formados(firma: str) -> None:
    assert not verificar_firma_callback(SECRETO, "1793016000", b"{}", firma)


# --- El envío -----------------------------------------------------------------------------------


@pytest.mark.requisito("RF-21")
async def test_envia_el_cuerpo_serializado_con_su_firma_y_su_timestamp() -> None:
    envio, _ = enviador()
    with respx.mock() as router:
        ruta = router.post(RUTA).mock(return_value=httpx.Response(200))
        await envio.enviar(URL, trabajo())

    pedido = ruta.calls.last.request
    assert pedido.content == serializar(trabajo())
    assert pedido.headers["content-type"] == "application/json"
    assert pedido.headers[ENCABEZADO_TIMESTAMP] == TIMESTAMP_T0
    assert pedido.headers[ENCABEZADO_FIRMA] == firma_independiente(SECRETO, TIMESTAMP_T0, pedido.content)


@pytest.mark.requisito("RF-21")
async def test_el_receptor_puede_verificar_lo_que_recibe_con_el_mismo_secreto() -> None:
    envio, _ = enviador()
    with respx.mock() as router:
        ruta = router.post(RUTA).mock(return_value=httpx.Response(200))
        await envio.enviar(URL, trabajo())
    pedido = ruta.calls.last.request
    assert verificar_firma_callback(
        SECRETO, pedido.headers[ENCABEZADO_TIMESTAMP], pedido.content, pedido.headers[ENCABEZADO_FIRMA]
    )


@pytest.mark.requisito("RF-21")
@pytest.mark.parametrize("estado", [200, 201, 202, 204])
async def test_cualquier_2xx_es_una_entrega_exitosa(estado: int) -> None:
    envio, _ = enviador()
    with respx.mock() as router:
        router.post(RUTA).mock(return_value=httpx.Response(estado))
        await envio.enviar(URL, trabajo())  # no lanza


@pytest.mark.requisito("RF-21")
@pytest.mark.parametrize("estado", [400, 401, 404, 429, 500, 503])
async def test_una_respuesta_que_no_es_2xx_es_callback_fallido(estado: int) -> None:
    envio, _ = enviador()
    with respx.mock() as router:
        router.post(RUTA).mock(return_value=httpx.Response(estado))
        with pytest.raises(CallbackFallido, match=str(estado)):
            await envio.enviar(URL, trabajo())


@pytest.mark.requisito("RF-21")
async def test_no_sigue_redirecciones_porque_reenviaria_el_cuerpo_firmado_a_otro_host() -> None:
    envio, _ = enviador()
    with respx.mock(assert_all_called=False) as router:
        router.post(RUTA).mock(return_value=httpx.Response(307, headers={"location": "https://interno.local/x"}))
        destino = router.post("https://interno.local/x").mock(return_value=httpx.Response(200))
        with pytest.raises(CallbackFallido):
            await envio.enviar(URL, trabajo())
        assert not destino.called


@pytest.mark.requisito("RF-21")
@pytest.mark.parametrize(
    "url",
    [
        "https://otro.example/cb",
        "http://backend.example/cb",
        "https://usuario:clave@backend.example/cb",
        "https://127.0.0.1/cb",
    ],
)
async def test_una_url_fuera_de_la_allowlist_no_hace_ninguna_peticion(url: str) -> None:
    envio, _ = enviador()
    with respx.mock(assert_all_called=False) as router:
        ruta = router.route().mock(return_value=httpx.Response(200))
        with pytest.raises(CallbackFallido):
            await envio.enviar(url, trabajo())
        assert not ruta.called


@pytest.mark.requisito("RF-21")
async def test_con_la_allowlist_vacia_no_se_envia_nada() -> None:
    envio, _ = enviador(hosts=[])
    with respx.mock(assert_all_called=False) as router:
        ruta = router.route().mock(return_value=httpx.Response(200))
        with pytest.raises(CallbackFallido):
            await envio.enviar(URL, trabajo())
        assert not ruta.called


@pytest.mark.requisito("RF-21")
@pytest.mark.parametrize("error", [httpx.ConnectTimeout("t"), httpx.ReadTimeout("t"), httpx.ConnectError("c")])
async def test_timeouts_y_errores_de_conexion_son_callback_fallido(error: httpx.HTTPError) -> None:
    envio, _ = enviador()
    with respx.mock() as router:
        router.post(RUTA).mock(side_effect=error)
        with pytest.raises(CallbackFallido):
            await envio.enviar(URL, trabajo())


@pytest.mark.requisito("RF-21")
async def test_ni_el_error_ni_el_log_incluyen_la_url_ni_el_secreto(caplog: pytest.LogCaptureFixture) -> None:
    envio, _ = enviador()
    with respx.mock() as router:
        router.post(RUTA).mock(side_effect=httpx.ConnectError("no conecta"))
        with caplog.at_level(logging.DEBUG), pytest.raises(CallbackFallido) as error:
            await envio.enviar(URL, trabajo())
    for texto in (str(error.value), caplog.text):
        assert "DATO-SENSIBLE" not in texto
        assert SECRETO not in texto


# --- Con los reintentos de NotificarResultado ---------------------------------------------------


class DormidorQueAvanzaElReloj(DormidorFalso):
    def __init__(self, reloj: RelojFijo) -> None:
        super().__init__()
        self._reloj = reloj

    async def dormir(self, segundos: float) -> None:
        await super().dormir(segundos)
        self._reloj.avanzar(segundos=segundos)


@pytest.mark.requisito("RF-21")
async def test_reintenta_con_backoff_y_cada_intento_lleva_su_propio_timestamp_y_firma_valida() -> None:
    envio, reloj = enviador()
    dormidor = DormidorQueAvanzaElReloj(reloj)
    caso = NotificarResultado(envio, dormidor, intentos=3, espera_base_s=1.0)

    with respx.mock() as router:
        ruta = router.post(RUTA).mock(side_effect=[httpx.Response(500), httpx.Response(503), httpx.Response(200)])
        assert await caso.ejecutar(trabajo())

    assert ruta.call_count == 3
    assert dormidor.esperas == [1.0, 2.0]
    timestamps = [c.request.headers[ENCABEZADO_TIMESTAMP] for c in ruta.calls]
    assert timestamps == [TIMESTAMP_T0, str(int(T0.timestamp()) + 1), str(int(T0.timestamp()) + 3)]
    for llamada in ruta.calls:  # un reintento no reusa la firma vieja
        pedido = llamada.request
        assert verificar_firma_callback(
            SECRETO, pedido.headers[ENCABEZADO_TIMESTAMP], pedido.content, pedido.headers[ENCABEZADO_FIRMA]
        )


@pytest.mark.requisito("RF-21")
async def test_si_el_backend_no_responde_nunca_se_agotan_los_intentos_y_no_se_lanza() -> None:
    envio, _ = enviador()
    caso = NotificarResultado(envio, DormidorFalso(), intentos=3, espera_base_s=1.0)
    with respx.mock() as router:
        ruta = router.post(RUTA).mock(return_value=httpx.Response(500))
        assert not await caso.ejecutar(trabajo())
        assert ruta.call_count == 3
