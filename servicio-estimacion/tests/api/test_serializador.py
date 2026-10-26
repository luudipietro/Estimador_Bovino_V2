import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from bv_estimacion.api.v1.esquemas.trabajo import serializar_trabajo
from bv_estimacion.dominio.errores import FALLAS_INTERNAS, CodigoError, MotivoNoSe
from bv_estimacion.dominio.estimacion import ResultadoFoto, Trabajo, agregar_resultados
from bv_estimacion.dominio.valores import IdEvento, IdTrabajo, OrigenImagen, ReferenciaImagen, VersionModelo

VERSION = VersionModelo("2026-09-25 / entrenado con AcmeAI-BMGF (Bangladesh)", 36.0, 621.0)
T0 = datetime(2026, 10, 25, 12, 0, 0, tzinfo=UTC)


def pendiente(metadatos: dict[str, object] | None = None) -> Trabajo:
    referencias = [ReferenciaImagen(OrigenImagen.ALMACEN_TEMPORAL, "tmp-1", "a" * 64)]
    return Trabajo.crear(IdTrabajo("trb_01"), IdEvento("evt-1"), referencias, T0, metadatos=metadatos)


def como_dict(trabajo: Trabajo) -> dict[str, Any]:
    cuerpo = json.loads(serializar_trabajo(trabajo))
    assert isinstance(cuerpo, dict)
    return cuerpo


def completado(*fotos: ResultadoFoto, metadatos: dict[str, object] | None = None) -> Trabajo:
    resultado = agregar_resultados(list(fotos), VERSION, ("la imagen 2 no trae metadata EXIF",))
    en_curso = pendiente(metadatos).iniciar(T0 + timedelta(seconds=1))
    return en_curso.completar(resultado, T0 + timedelta(seconds=3), descarga_ms=100, inferencia_ms=1840)


@pytest.mark.requisito("RF-19")
def test_un_trabajo_pendiente_dice_solo_quien_es_y_en_que_estado_esta() -> None:
    assert como_dict(pendiente()) == {
        "id": "trb_01",
        "id_evento": "evt-1",
        "estado": "pendiente",
        "creado": "2026-10-25T12:00:00Z",
    }


@pytest.mark.requisito("RF-19")
def test_un_trabajo_procesando_se_serializa_igual_que_uno_pendiente_con_su_estado() -> None:
    cuerpo = como_dict(pendiente().iniciar(T0 + timedelta(seconds=1)))
    assert cuerpo["estado"] == "procesando"
    assert set(cuerpo) == {"id", "id_evento", "estado", "creado"}


@pytest.mark.requisito("RF-20")
def test_un_trabajo_completado_trae_todo_lo_del_contrato() -> None:
    trabajo = completado(
        ResultadoFoto(185.9, True, (), 0.94, 0.88, 1),
        ResultadoFoto(190.0, True, (), 0.91, 0.80, 2),
        metadatos={"raza": "Brangus"},
    )

    cuerpo = como_dict(trabajo)

    assert set(cuerpo) == {
        "id", "id_evento", "estado", "peso_kg", "confiable", "motivos", "version_modelo", "rango_validez_kg",
        "imagenes", "advertencias", "metadatos", "duracion_ms", "creado", "finalizado",
    }  # fmt: skip
    assert cuerpo["estado"] == "completado"
    assert cuerpo["peso_kg"] == 187.9  # mediana de 185.9 y 190.0
    assert cuerpo["confiable"] is True
    assert cuerpo["motivos"] == []
    assert cuerpo["version_modelo"] == VERSION.version  # R-23
    assert cuerpo["rango_validez_kg"] == [36.0, 621.0]
    assert cuerpo["advertencias"] == ["la imagen 2 no trae metadata EXIF"]
    assert cuerpo["duracion_ms"] == {"cola": 1000, "inferencia": 1840, "total": 3000}
    assert cuerpo["creado"] == "2026-10-25T12:00:00Z"
    assert cuerpo["finalizado"] == "2026-10-25T12:00:03Z"


@pytest.mark.requisito("RF-20")
def test_el_detalle_de_cada_foto_va_numerado_desde_1_en_el_orden_del_pedido() -> None:
    cuerpo = como_dict(completado(ResultadoFoto(185.9, True, (), 0.94, 0.88, 1), ResultadoFoto(190.0, True)))
    assert cuerpo["imagenes"] == [
        {
            "indice": 1, "peso_kg": 185.9, "confiable": True, "motivos": [],
            "confianza_deteccion": 0.94, "confianza_keypoint_minima": 0.88, "bovinos_en_la_escena": 1,
        },
        {
            "indice": 2, "peso_kg": 190.0, "confiable": True, "motivos": [],
            "confianza_deteccion": None, "confianza_keypoint_minima": None, "bovinos_en_la_escena": None,
        },
    ]  # fmt: skip


@pytest.mark.requisito("R-21")
def test_r21_el_no_se_es_un_completado_con_peso_nulo_y_sus_motivos() -> None:
    sin_animal = ResultadoFoto(None, False, (MotivoNoSe.SIN_ANIMAL,), bovinos_en_la_escena=0)

    cuerpo = como_dict(completado(sin_animal))

    assert cuerpo["estado"] == "completado"
    assert cuerpo["confiable"] is False
    assert "peso_kg" in cuerpo and cuerpo["peso_kg"] is None  # el campo está, con null
    assert cuerpo["motivos"] == ["sin_animal"]
    assert cuerpo["imagenes"][0]["motivos"] == ["sin_animal"]


@pytest.mark.requisito("R-20")
def test_r20_un_peso_fuera_de_rango_sale_no_confiable_con_ese_motivo() -> None:
    cuerpo = como_dict(completado(ResultadoFoto(700.0, True)))
    assert cuerpo["confiable"] is False
    assert cuerpo["motivos"] == ["fuera_de_rango"]
    assert cuerpo["peso_kg"] == 700.0


@pytest.mark.requisito("R-26")
def test_r26_los_metadatos_vuelven_sin_modificar() -> None:
    metadatos = {"raza": "Brangus", "lote": {"id": 7, "tags": ["a", "b"]}, "peso_declarado": None, "n": 3.5}
    assert como_dict(completado(ResultadoFoto(150.0, True), metadatos=metadatos))["metadatos"] == metadatos


@pytest.mark.requisito("RF-22")
def test_un_trabajo_fallido_trae_codigo_detalle_generico_y_si_se_puede_reintentar() -> None:
    trabajo = pendiente().iniciar(T0).fallar(CodigoError.TIEMPO_EXCEDIDO, T0 + timedelta(seconds=45), reintentable=True)

    assert como_dict(trabajo) == {
        "id": "trb_01",
        "id_evento": "evt-1",
        "estado": "fallido",
        "codigo": "tiempo_excedido",
        "detalle": "La estimación superó el presupuesto de tiempo.",
        "reintentable": True,
        "creado": "2026-10-25T12:00:00Z",
        "finalizado": "2026-10-25T12:00:45Z",
    }


@pytest.mark.requisito("RF-22")
@pytest.mark.parametrize("codigo", sorted(FALLAS_INTERNAS))
def test_toda_falla_interna_tiene_un_detalle_legible_sin_trazas_ni_nombres_de_clase(codigo: CodigoError) -> None:
    trabajo = pendiente().iniciar(T0).fallar(codigo, T0, reintentable=False)

    detalle = como_dict(trabajo)["detalle"]

    assert detalle and detalle[0].isupper() and detalle.endswith(".")
    for pista in ("Traceback", "Error:", "Exception", ".py", "bv_estimacion"):
        assert pista not in detalle


@pytest.mark.requisito("RF-21")
def test_el_cuerpo_es_json_utf8_y_es_el_mismo_cada_vez() -> None:
    """El `GET` y el callback usan estos bytes: tienen que ser idénticos para el mismo trabajo."""
    trabajo = completado(ResultadoFoto(150.0, True), metadatos={"nota": "año de ñandú"})

    primero, segundo = serializar_trabajo(trabajo), serializar_trabajo(trabajo)

    assert primero == segundo
    assert isinstance(primero, bytes)
    assert json.loads(primero.decode("utf-8"))["metadatos"]["nota"] == "año de ñandú"
