import json
import threading
from pathlib import Path

import anyio
import httpx
import pytest
import respx

from tests import imagenes_sinteticas as img
from tests.api.ayudas import (
    AUTORIZACION,
    RUTA,
    archivos,
    encabezados,
    esperar_estado,
    foto,
    pedir,
    servicio,
    sha256,
)
from tests.modelo_falso import SECRETO, FabricaFalsa, respuesta_modelo

URL_IMAGEN = "https://almacenamiento.example/fotos/animal.jpg?firma=FIRMA-SECRETA"
URL_CALLBACK = "https://backend.example/api/estimaciones/callback"


# --- El camino feliz ----------------------------------------------------------------------------


@pytest.mark.requisito("RF-19")
async def test_un_pedido_por_multipart_responde_202_con_location_y_queda_en_cola(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await pedir(cliente, foto(), id_evento="evt-1")

    assert respuesta.status_code == 202
    cuerpo = respuesta.json()
    assert set(cuerpo) == {"id", "id_evento", "estado"}
    assert cuerpo["id"].startswith("trb_")
    assert cuerpo["id_evento"] == "evt-1"
    assert cuerpo["estado"] == "pendiente"
    assert respuesta.headers["location"] == f"/v1/estimaciones/{cuerpo['id']}"


@pytest.mark.requisito("RF-20")
async def test_el_trabajo_se_procesa_y_el_get_devuelve_el_resultado_completo(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        aceptado = (await pedir(cliente, foto(), metadatos={"raza": "Brangus"})).json()

        respuesta = await esperar_estado(cliente, aceptado["id"])

    cuerpo = respuesta.json()
    assert respuesta.status_code == 200
    assert cuerpo["estado"] == "completado"
    assert cuerpo["peso_kg"] == 187.4
    assert cuerpo["confiable"] is True
    assert cuerpo["motivos"] == []
    assert cuerpo["version_modelo"] == "v-prueba / datos de prueba"
    assert cuerpo["rango_validez_kg"] == [36.0, 621.0]
    assert [i["indice"] for i in cuerpo["imagenes"]] == [1]
    assert cuerpo["metadatos"] == {"raza": "Brangus"}  # R-26
    assert set(cuerpo["duracion_ms"]) == {"cola", "inferencia", "total"}


@pytest.mark.requisito("RF-20")
async def test_con_varias_fotos_el_detalle_va_numerado_en_el_orden_del_pedido(tmp_path: Path) -> None:
    fabrica = FabricaFalsa()
    primera, segunda, tercera = foto(1), foto(2), foto(3)
    fabrica.respuestas = {
        primera: respuesta_modelo(100.0),
        segunda: respuesta_modelo(200.0),
        tercera: respuesta_modelo(150.0),
    }
    async with servicio(tmp_path, fabrica=fabrica) as (_, cliente):
        aceptado = (await pedir(cliente, primera, segunda, tercera)).json()
        cuerpo = (await esperar_estado(cliente, aceptado["id"])).json()

    assert [i["peso_kg"] for i in cuerpo["imagenes"]] == [100.0, 200.0, 150.0]
    assert cuerpo["peso_kg"] == 150.0  # R-22: la mediana de las confiables


@pytest.mark.requisito("R-21")
async def test_el_no_se_es_un_completado_con_peso_nulo_y_no_un_error(tmp_path: Path) -> None:
    fabrica = FabricaFalsa()
    fabrica.respuesta_por_defecto = {
        "peso_kg": None,
        "confiable": False,
        "motivo": "no se detectó ningún bovino en la foto",
    }
    async with servicio(tmp_path, fabrica=fabrica) as (_, cliente):
        aceptado = (await pedir(cliente, foto())).json()
        respuesta = await esperar_estado(cliente, aceptado["id"])

    assert respuesta.status_code == 200
    assert respuesta.json()["confiable"] is False
    assert respuesta.json()["peso_kg"] is None
    assert respuesta.json()["motivos"] == ["sin_animal"]


@pytest.mark.requisito("RF-22")
async def test_una_falla_del_modelo_deja_el_trabajo_fallido_con_detalle_generico(tmp_path: Path) -> None:
    fabrica = FabricaFalsa()
    fabrica.error_al_estimar = RuntimeError("detalle interno /srv/app/modelo.py")
    async with servicio(tmp_path, fabrica=fabrica) as (_, cliente):
        aceptado = (await pedir(cliente, foto())).json()
        respuesta = await esperar_estado(cliente, aceptado["id"], "fallido")

    cuerpo = respuesta.json()
    assert respuesta.status_code == 200  # la consulta salió bien; el trabajo es el que falló
    assert cuerpo["codigo"] == "error_interno"
    assert cuerpo["reintentable"] is False
    assert "RuntimeError" not in respuesta.text and "/srv/app" not in respuesta.text


@pytest.mark.requisito("RF-19")
async def test_mientras_el_modelo_trabaja_el_get_dice_procesando(tmp_path: Path) -> None:
    fabrica = FabricaFalsa()
    fabrica.frenar = threading.Event()
    async with servicio(tmp_path, fabrica=fabrica) as (_, cliente):
        aceptado = (await pedir(cliente, foto())).json()
        en_curso = await esperar_estado(cliente, aceptado["id"], "procesando")
        assert set(en_curso.json()) == {"id", "id_evento", "estado", "creado"}
        fabrica.frenar.set()
        await esperar_estado(cliente, aceptado["id"])


@pytest.mark.requisito("RF-22")
async def test_un_id_en_blanco_es_404_y_no_un_error_interno(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.get(f"{RUTA}/%20", headers=AUTORIZACION)
    assert respuesta.status_code == 404
    assert respuesta.json()["codigo"] == "trabajo_no_encontrado"


@pytest.mark.requisito("RF-22")
async def test_un_trabajo_que_no_existe_es_404_trabajo_no_encontrado(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.get(f"{RUTA}/trb_NOEXISTE", headers=AUTORIZACION)
    assert respuesta.status_code == 404
    assert respuesta.json()["codigo"] == "trabajo_no_encontrado"


# --- Por referencia (URL firmada) -----------------------------------------------------------------


@pytest.mark.requisito("RF-19")
async def test_un_pedido_por_referencia_descarga_verifica_el_sha256_y_se_procesa(tmp_path: Path) -> None:
    datos = foto()
    with respx.mock(assert_all_called=False) as router:
        descarga = router.get(URL_IMAGEN).mock(return_value=httpx.Response(200, content=datos))
        async with servicio(tmp_path) as (_, cliente):
            respuesta = await cliente.post(
                RUTA,
                headers=encabezados("evt-url"),
                json={"referencias_imagen": [{"url": URL_IMAGEN, "sha256": sha256(datos)}]},
            )
            assert respuesta.status_code == 202
            final = await esperar_estado(cliente, respuesta.json()["id"])

    assert descarga.called
    assert final.json()["peso_kg"] == 187.4


@pytest.mark.requisito("RF-22")
async def test_si_el_contenido_no_coincide_con_el_sha256_declarado_el_trabajo_queda_referencia_inaccesible(
    tmp_path: Path,
) -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(URL_IMAGEN).mock(return_value=httpx.Response(200, content=foto()))
        async with servicio(tmp_path) as (_, cliente):
            respuesta = await cliente.post(
                RUTA,
                headers=encabezados("evt-url"),
                json={"referencias_imagen": [{"url": URL_IMAGEN, "sha256": "a" * 64}]},
            )
            final = await esperar_estado(cliente, respuesta.json()["id"], "fallido")

    assert final.json()["codigo"] == "referencia_inaccesible"
    assert final.json()["reintentable"] is True
    assert "FIRMA-SECRETA" not in final.text


@pytest.mark.requisito("RF-19")
async def test_el_sha256_en_mayusculas_es_el_mismo_pedido_que_en_minusculas(tmp_path: Path) -> None:
    datos = foto()
    cuerpo = {"referencias_imagen": [{"url": URL_IMAGEN, "sha256": sha256(datos)}]}
    mayusculas = {"referencias_imagen": [{"url": URL_IMAGEN, "sha256": sha256(datos).upper()}]}
    with respx.mock(assert_all_called=False) as router:
        router.get(URL_IMAGEN).mock(return_value=httpx.Response(200, content=datos))
        async with servicio(tmp_path) as (_, cliente):
            primero = await cliente.post(RUTA, headers=encabezados("evt-1"), json=cuerpo)
            segundo = await cliente.post(RUTA, headers=encabezados("evt-1"), json=mayusculas)

    assert (primero.status_code, segundo.status_code) == (202, 200)
    assert primero.json()["id"] == segundo.json()["id"]


# --- Idempotencia ----------------------------------------------------------------------------------


@pytest.mark.requisito("RF-19")
async def test_el_mismo_pedido_otra_vez_devuelve_200_con_el_mismo_trabajo(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        primero = await pedir(cliente, foto(), id_evento="evt-1")
        segundo = await pedir(cliente, foto(), id_evento="evt-1")

    assert (primero.status_code, segundo.status_code) == (202, 200)
    assert segundo.json()["id"] == primero.json()["id"]
    assert segundo.headers["location"] == primero.headers["location"]


@pytest.mark.requisito("RF-19")
async def test_el_mismo_pedido_ya_terminado_devuelve_el_trabajo_con_su_estado_actual(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        primero = (await pedir(cliente, foto())).json()
        await esperar_estado(cliente, primero["id"])
        repetido = await pedir(cliente, foto())

    assert repetido.status_code == 200
    assert repetido.json() == {"id": primero["id"], "id_evento": "evt-1", "estado": "completado"}


@pytest.mark.requisito("RF-17")
async def test_las_mismas_fotos_en_otro_orden_son_el_mismo_pedido(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        primero = await pedir(cliente, foto(1), foto(2), id_evento="evt-1")
        invertido = await pedir(cliente, foto(2), foto(1), id_evento="evt-1")

    assert (primero.status_code, invertido.status_code) == (202, 200)
    assert invertido.json()["id"] == primero.json()["id"]


@pytest.mark.requisito("RF-19")
async def test_el_mismo_id_evento_con_otra_imagen_es_409(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        await pedir(cliente, foto(1), id_evento="evt-1")
        conflicto = await pedir(cliente, foto(2), id_evento="evt-1")

    assert conflicto.status_code == 409
    assert conflicto.json()["codigo"] == "id_evento_en_conflicto"


@pytest.mark.requisito("RF-19")
async def test_ids_de_evento_distintos_son_trabajos_distintos(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        uno = await pedir(cliente, foto(), id_evento="evt-1")
        dos = await pedir(cliente, foto(), id_evento="evt-2")
    assert uno.json()["id"] != dos.json()["id"]


# --- Idempotency-Key -------------------------------------------------------------------------------


@pytest.mark.requisito("RF-19")
@pytest.mark.parametrize("clave", [None, "", "   ", "con espacios", "x" * 129, "salto\ttab"])
async def test_sin_una_idempotency_key_valida_el_pedido_es_solicitud_invalida(
    tmp_path: Path, clave: str | None
) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.post(RUTA, headers=encabezados(clave), files=archivos(foto()))

    assert respuesta.status_code == 400
    assert respuesta.json()["codigo"] == "solicitud_invalida"
    assert "Idempotency-Key" in respuesta.json()["detail"]


@pytest.mark.requisito("RF-19")
async def test_la_idempotency_key_tiene_que_venir_en_el_encabezado_no_en_el_cuerpo(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.post(RUTA, headers=AUTORIZACION, files=archivos(foto()), data={"id_evento": "evt-1"})
    assert respuesta.status_code == 400


# --- Cantidad y forma del pedido -----------------------------------------------------------------


@pytest.mark.requisito("RF-19")
async def test_sin_imagenes_es_422_cantidad_imagenes_invalida(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        # `(None, valor)` hace que httpx mande un multipart aunque no haya archivos
        respuesta = await cliente.post(RUTA, headers=encabezados(), files={"metadatos": (None, "{}")})
    assert respuesta.status_code == 422
    assert respuesta.json()["codigo"] == "cantidad_imagenes_invalida"


@pytest.mark.requisito("RF-19")
async def test_cinco_imagenes_es_422_y_cuatro_se_aceptan(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        cuatro = await pedir(cliente, *(foto(i) for i in range(4)), id_evento="evt-4")
        cinco = await pedir(cliente, *(foto(i) for i in range(5)), id_evento="evt-5")

    assert cuatro.status_code == 202
    assert cinco.status_code == 422
    assert cinco.json()["codigo"] == "cantidad_imagenes_invalida"


@pytest.mark.requisito("RF-19")
async def test_imagenes_y_referencias_a_la_vez_es_422(tmp_path: Path) -> None:
    referencias = json.dumps([{"url": URL_IMAGEN, "sha256": "a" * 64}])
    async with servicio(tmp_path) as (_, cliente):
        por_multipart = await cliente.post(
            RUTA, headers=encabezados(), files=archivos(foto()), data={"referencias_imagen": referencias}
        )
        por_json = await cliente.post(
            RUTA, headers=encabezados("evt-2"), json={"imagenes": ["x"], "referencias_imagen": json.loads(referencias)}
        )

    for respuesta in (por_multipart, por_json):
        assert respuesta.status_code == 422
        assert respuesta.json()["codigo"] == "cantidad_imagenes_invalida"


@pytest.mark.requisito("RF-19")
@pytest.mark.parametrize("cantidad", [0, 5])
async def test_una_cantidad_de_referencias_fuera_de_1_a_4_es_422(tmp_path: Path, cantidad: int) -> None:
    referencias = [{"url": URL_IMAGEN, "sha256": f"{i:064x}"} for i in range(cantidad)]
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.post(RUTA, headers=encabezados(), json={"referencias_imagen": referencias})
    assert respuesta.status_code == 422
    assert respuesta.json()["codigo"] == "cantidad_imagenes_invalida"


@pytest.mark.requisito("RF-19")
@pytest.mark.parametrize("referencias", ["no es json", '{"url": "x"}', '[{"url": "https://almacenamiento.example/a"}]'])
async def test_referencias_imagen_mal_formado_en_multipart_es_solicitud_invalida(
    tmp_path: Path, referencias: str
) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.post(RUTA, headers=encabezados(), files={"referencias_imagen": (None, referencias)})
    assert respuesta.status_code == 400
    assert respuesta.json()["campos"] == ["referencias_imagen"]


@pytest.mark.requisito("RF-19")
async def test_un_campo_desconocido_en_multipart_es_solicitud_invalida(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.post(
            RUTA, headers=encabezados(), files=archivos(foto()), data={"url_notificacon": URL_CALLBACK}
        )
    assert respuesta.status_code == 400
    assert respuesta.json()["codigo"] == "solicitud_invalida"
    assert "url_notificacon" in respuesta.json()["campos"]


@pytest.mark.requisito("RF-19")
async def test_un_campo_repetido_en_multipart_es_solicitud_invalida(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.post(
            RUTA, headers=encabezados(), files=archivos(foto()), data={"metadatos": ["{}", "{}"]}
        )
    assert respuesta.status_code == 400


@pytest.mark.requisito("RF-19")
async def test_un_archivo_en_un_campo_de_texto_o_texto_en_el_campo_de_imagenes_es_solicitud_invalida(
    tmp_path: Path,
) -> None:
    async with servicio(tmp_path) as (_, cliente):
        archivo_en_metadatos = await cliente.post(
            RUTA, headers=encabezados(), files=archivos(foto()) + archivos(b"{}", campo="metadatos")
        )
        texto_en_imagenes = await cliente.post(
            RUTA, headers=encabezados("evt-2"), files={"imagenes": (None, "no soy un archivo")}
        )
    assert archivo_en_metadatos.status_code == 400
    assert texto_en_imagenes.status_code == 400


@pytest.mark.requisito("RF-19")
async def test_un_content_type_que_no_es_multipart_ni_json_es_solicitud_invalida(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.post(RUTA, headers=encabezados() | {"Content-Type": "text/plain"}, content=b"hola")
    assert respuesta.status_code == 400
    assert respuesta.json()["codigo"] == "solicitud_invalida"


@pytest.mark.requisito("RF-19")
@pytest.mark.parametrize(
    "cuerpo",
    [
        b"{ esto no es json",
        b"[]",  # JSON válido pero no un objeto
        b'{"referencias_imagen": "no es una lista"}',
        b'{"referencias_imagen": [{"url": "https://almacenamiento.example/a.jpg"}]}',  # falta sha256
        b'{"referencias_imagen": [{"url": "https://almacenamiento.example/a.jpg", "sha256": "corto"}]}',
        b'{"referencias_imagen": [{"url": "https://almacenamiento.example/a.jpg", "sha256": "z' + b"z" * 63 + b'"}]}',
        b'{"referencias_imagen": [], "campo_raro": 1}',
        b'{"referencias_imagen": [], "metadatos": [1, 2]}',
        b'{"referencias_imagen": [], "url_notificacion": 5}',
    ],
)
async def test_un_json_mal_formado_es_solicitud_invalida(tmp_path: Path, cuerpo: bytes) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.post(
            RUTA, headers=encabezados() | {"Content-Type": "application/json"}, content=cuerpo
        )
    assert respuesta.status_code == 400, respuesta.text
    assert respuesta.json()["codigo"] == "solicitud_invalida"


@pytest.mark.requisito("R-26")
@pytest.mark.parametrize("metadatos", ["[1, 2]", '"texto"', "7", "{ no es json"])
async def test_metadatos_que_no_son_un_objeto_json_son_solicitud_invalida(tmp_path: Path, metadatos: str) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.post(
            RUTA, headers=encabezados(), files=archivos(foto()), data={"metadatos": metadatos}
        )
    assert respuesta.status_code == 400
    assert respuesta.json()["codigo"] == "solicitud_invalida"


@pytest.mark.requisito("R-26")
async def test_los_metadatos_vuelven_tal_cual_y_no_cambian_la_estimacion(tmp_path: Path) -> None:
    metadatos = {"raza": "Brangus", "lote": {"id": 7, "tags": ["a", "b"]}, "peso_declarado": None}
    async with servicio(tmp_path) as (_, cliente):
        con = (await pedir(cliente, foto(), id_evento="evt-1", metadatos=metadatos)).json()
        sin = (await pedir(cliente, foto(), id_evento="evt-2")).json()
        con_final = (await esperar_estado(cliente, con["id"])).json()
        sin_final = (await esperar_estado(cliente, sin["id"])).json()

    assert con_final["metadatos"] == metadatos
    assert sin_final["metadatos"] == {}
    assert con_final["peso_kg"] == sin_final["peso_kg"]


# --- Validación de las imágenes en el borde ---------------------------------------------------------


@pytest.mark.requisito("RF-15")
@pytest.mark.parametrize(
    ("datos", "estado", "codigo"),
    [
        pytest.param(img.gif(), 415, "formato_no_admitido", id="gif"),
        pytest.param(img.bmp(), 415, "formato_no_admitido", id="bmp"),
        pytest.param(b"%PDF-1.4 esto no es una imagen", 415, "formato_no_admitido", id="pdf"),
        pytest.param(img.jpeg(1024, 768)[:200], 422, "imagen_corrupta", id="jpeg-truncado"),
        pytest.param(b"", 422, "imagen_corrupta", id="vacio"),
        pytest.param(img.jpeg(800, 600), 422, "resolucion_insuficiente", id="800x600"),  # 480 000 px < 0,75 MP
        pytest.param(img.jpeg(1023, 733), 422, "resolucion_insuficiente", id="justo-por-debajo"),  # 749 859 px
    ],
)
async def test_una_imagen_invalida_se_rechaza_al_recibir_con_su_codigo(
    tmp_path: Path, datos: bytes, estado: int, codigo: str
) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await pedir(cliente, datos)

    assert respuesta.status_code == estado
    assert respuesta.json()["codigo"] == codigo


@pytest.mark.requisito("RF-15")
async def test_un_gif_con_nombre_y_content_type_de_jpeg_se_rechaza_porque_se_miran_los_magic_bytes(
    tmp_path: Path,
) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.post(
            RUTA,
            headers=encabezados(),
            files=[("imagenes", ("foto.jpg", img.gif(), "image/jpeg"))],
        )
    assert respuesta.status_code == 415


@pytest.mark.requisito("RF-15")
async def test_la_resolucion_justo_en_el_minimo_se_acepta(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await pedir(cliente, img.jpeg(1000, 750))  # 750 000 px = 0,75 MP
    assert respuesta.status_code == 202


@pytest.mark.requisito("R-08")
async def test_una_imagen_heic_se_acepta_y_se_procesa(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await cliente.post(
            RUTA,
            headers=encabezados(),
            files=[("imagenes", ("foto.heic", img.heic(1024, 768), "image/heic"))],
        )
        assert respuesta.status_code == 202
        await esperar_estado(cliente, respuesta.json()["id"])


@pytest.mark.requisito("RF-15")
async def test_una_imagen_que_supera_el_maximo_de_mb_es_413_imagen_muy_grande(tmp_path: Path) -> None:
    grande = img.jpeg(2400, 2000, ruido=True)  # ruido incompresible: casi 3 MB
    assert len(grande) > 1024 * 1024
    async with servicio(tmp_path, max_mb_imagen=1) as (_, cliente):
        respuesta = await pedir(cliente, grande)

    assert respuesta.status_code == 413
    assert respuesta.json()["codigo"] == "imagen_muy_grande"


@pytest.mark.requisito("RF-15")
async def test_una_imagen_con_mas_pixeles_que_el_limite_es_413_imagen_muy_grande(tmp_path: Path) -> None:
    async with servicio(tmp_path, max_pixeles=500_000) as (_, cliente):
        respuesta = await pedir(cliente, foto())  # 786 432 px
    assert respuesta.status_code == 413
    assert respuesta.json()["codigo"] == "imagen_muy_grande"


@pytest.mark.requisito("RF-15")
async def test_un_pedido_con_una_imagen_mala_no_deja_trabajo_ni_archivos_temporales(tmp_path: Path) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await pedir(cliente, foto(1), img.gif(), id_evento="evt-1")
        assert respuesta.status_code == 415
        # La buena de las dos no se guardó: se valida todo antes de tocar el disco.
        assert not (tmp_path / "temporal").exists() or not list((tmp_path / "temporal").iterdir())
        repetido = await pedir(cliente, foto(1), id_evento="evt-1")  # el id_evento quedó libre

    assert repetido.status_code == 202


# --- Allowlists ---------------------------------------------------------------------------------------


@pytest.mark.requisito("RF-22")
@pytest.mark.parametrize(
    "url",
    [
        "https://otro-host.example/cb",
        "http://backend.example/cb",
        "https://usuario:clave@backend.example/cb",
        "https://127.0.0.1/cb",
        "ftp://backend.example/cb",
    ],
)
async def test_una_url_de_notificacion_fuera_de_la_allowlist_es_422_y_no_repite_la_url(
    tmp_path: Path, url: str
) -> None:
    async with servicio(tmp_path) as (_, cliente):
        respuesta = await pedir(cliente, foto(), url_notificacion=url)

    assert respuesta.status_code == 422
    assert respuesta.json()["codigo"] == "referencia_no_permitida"
    assert url not in respuesta.text


@pytest.mark.requisito("RF-22")
@pytest.mark.parametrize(
    "url", ["https://otro-host.example/f.jpg?firma=SECRETA", "http://almacenamiento.example/f.jpg"]
)
async def test_una_referencia_de_imagen_fuera_de_la_allowlist_es_422_y_no_se_descarga(tmp_path: Path, url: str) -> None:
    with respx.mock(assert_all_called=False) as router:
        ruta = router.route().mock(return_value=httpx.Response(200, content=foto()))
        async with servicio(tmp_path) as (_, cliente):
            respuesta = await cliente.post(
                RUTA, headers=encabezados(), json={"referencias_imagen": [{"url": url, "sha256": "a" * 64}]}
            )

    assert respuesta.status_code == 422
    assert respuesta.json()["codigo"] == "referencia_no_permitida"
    assert "SECRETA" not in respuesta.text
    assert not ruta.called


@pytest.mark.requisito("RF-22")
async def test_con_las_allowlists_vacias_no_se_acepta_ninguna_url(tmp_path: Path) -> None:
    async with servicio(tmp_path, hosts_imagen_permitidos=[], hosts_callback_permitidos=[]) as (_, cliente):
        callback = await pedir(cliente, foto(), url_notificacion=URL_CALLBACK)
        referencia = await cliente.post(
            RUTA, headers=encabezados("evt-2"), json={"referencias_imagen": [{"url": URL_IMAGEN, "sha256": "a" * 64}]}
        )
    assert callback.status_code == 422
    assert referencia.status_code == 422


# --- Callback ---------------------------------------------------------------------------------------


@pytest.mark.requisito("RF-21")
async def test_el_callback_lleva_exactamente_el_mismo_cuerpo_que_el_get_y_va_firmado(tmp_path: Path) -> None:
    from bv_estimacion.infraestructura.notificacion.callback_https import (
        ENCABEZADO_FIRMA,
        ENCABEZADO_TIMESTAMP,
        verificar_firma_callback,
    )

    with respx.mock(assert_all_called=False) as router:
        callback = router.post(URL_CALLBACK).mock(return_value=httpx.Response(200))
        async with servicio(tmp_path) as (_, cliente):
            aceptado = (await pedir(cliente, foto(), url_notificacion=URL_CALLBACK)).json()
            por_get = await esperar_estado(cliente, aceptado["id"])
            with anyio.fail_after(5):  # el aviso se envía en una tarea aparte, después de guardar el estado final
                while not callback.called:
                    await anyio.sleep(0.01)

    assert callback.called
    pedido = callback.calls.last.request
    assert pedido.content == por_get.content  # los mismos bytes, no solo el mismo JSON
    assert verificar_firma_callback(
        SECRETO, pedido.headers[ENCABEZADO_TIMESTAMP], pedido.content, pedido.headers[ENCABEZADO_FIRMA]
    )


@pytest.mark.requisito("RF-21")
async def test_si_el_backend_no_recibe_el_callback_el_resultado_sigue_disponible_por_get(tmp_path: Path) -> None:
    with respx.mock(assert_all_called=False) as router:
        router.post(URL_CALLBACK).mock(return_value=httpx.Response(500))
        async with servicio(tmp_path) as (_, cliente):
            aceptado = (await pedir(cliente, foto(), url_notificacion=URL_CALLBACK)).json()
            final = await esperar_estado(cliente, aceptado["id"])
    assert final.json()["estado"] == "completado"


# --- Cola llena ---------------------------------------------------------------------------------------


@pytest.mark.requisito("RF-19")
async def test_con_la_cola_llena_el_pedido_es_503_cola_llena_y_no_deja_nada_guardado(tmp_path: Path) -> None:
    fabrica = FabricaFalsa()
    fabrica.frenar = threading.Event()
    async with servicio(tmp_path, fabrica=fabrica, tamano_cola=1, hilos_inferencia=1) as (_, cliente):
        primero = (await pedir(cliente, foto(1), id_evento="evt-1")).json()
        await esperar_estado(cliente, primero["id"], "procesando")  # el worker ya lo tiene: la cola quedó vacía

        segundo = await pedir(cliente, foto(2), id_evento="evt-2")  # ocupa el único lugar
        tercero = await pedir(cliente, foto(3), id_evento="evt-3")  # no hay lugar

        assert segundo.status_code == 202
        assert tercero.status_code == 503
        assert tercero.json()["codigo"] == "cola_llena"
        assert int(tercero.headers["retry-after"]) > 0

        fabrica.frenar.set()
        await esperar_estado(cliente, segundo.json()["id"])
        reintento = await pedir(cliente, foto(3), id_evento="evt-3")  # con lugar, el mismo pedido entra
        assert reintento.status_code == 202


# --- Qué se puede ver ---------------------------------------------------------------------------------


@pytest.mark.requisito("RF-22")
async def test_ninguna_respuesta_repite_el_token_ni_la_url_firmada(tmp_path: Path) -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(URL_IMAGEN).mock(return_value=httpx.Response(404))
        async with servicio(tmp_path) as (_, cliente):
            respuesta = await cliente.post(
                RUTA, headers=encabezados(), json={"referencias_imagen": [{"url": URL_IMAGEN, "sha256": "a" * 64}]}
            )
            final = await esperar_estado(cliente, respuesta.json()["id"], "fallido")

    for texto in (respuesta.text, final.text, json.dumps(dict(final.headers))):
        assert "FIRMA-SECRETA" not in texto
        assert AUTORIZACION["Authorization"] not in texto
