import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio, MotivoNoSe, TransicionInvalida
from bv_estimacion.dominio.estimacion import (
    EstadoTrabajo,
    ResultadoEstimacion,
    ResultadoFoto,
    Trabajo,
    agregar_resultados,
    huella_de_imagenes,
)
from bv_estimacion.dominio.valores import IdEvento, IdTrabajo, VersionModelo

VERSION = VersionModelo("2026-09-25 / AcmeAI-BMGF", 36.0, 621.0)
T0 = datetime(2026, 10, 25, 12, 0, 0, tzinfo=UTC)


def foto(peso: float | None, *motivos: MotivoNoSe) -> ResultadoFoto:
    return ResultadoFoto(peso_kg=peso, confiable=not motivos and peso is not None, motivos=motivos)


# --- ResultadoFoto / ResultadoEstimacion --------------------------------------------------------


@pytest.mark.requisito("RF-20")
def test_un_resultado_confiable_necesita_peso_y_no_puede_tener_motivos() -> None:
    with pytest.raises(ValueError):
        ResultadoFoto(peso_kg=None, confiable=True)
    with pytest.raises(ValueError):
        ResultadoFoto(peso_kg=150.0, confiable=True, motivos=(MotivoNoSe.IMAGEN_BORROSA,))


@pytest.mark.requisito("RF-20")
def test_r21_un_resultado_no_confiable_siempre_dice_por_que() -> None:
    with pytest.raises(ValueError):
        ResultadoFoto(peso_kg=150.0, confiable=False)
    with pytest.raises(ValueError):
        ResultadoEstimacion(peso_kg=None, confiable=False, motivos=(), version=VERSION)


# --- agregar_resultados -------------------------------------------------------------------------


@pytest.mark.requisito("RF-20")
def test_r22_una_sola_foto_confiable_da_su_peso() -> None:
    resultado = agregar_resultados([foto(187.4)], VERSION)
    assert resultado.peso_kg == 187.4
    assert resultado.confiable
    assert resultado.motivos == ()


@pytest.mark.requisito("RF-20")
def test_r22_agrega_con_la_mediana_de_las_confiables() -> None:
    resultado = agregar_resultados([foto(180.0), foto(190.0), foto(400.0)], VERSION)
    assert resultado.peso_kg == 190.0


@pytest.mark.requisito("RF-20")
def test_r22_ignora_las_fotos_no_confiables_al_calcular_la_mediana() -> None:
    resultado = agregar_resultados([foto(180.0), foto(500.0, MotivoNoSe.IMAGEN_BORROSA), foto(190.0)], VERSION)
    assert resultado.peso_kg == 185.0
    assert resultado.confiable


@pytest.mark.requisito("RF-20")
def test_r22_si_ninguna_es_confiable_el_agregado_tampoco() -> None:
    resultado = agregar_resultados(
        [foto(None, MotivoNoSe.SIN_ANIMAL), foto(170.0, MotivoNoSe.PARTES_TAPADAS)],
        VERSION,
    )
    assert not resultado.confiable
    assert resultado.motivos == (MotivoNoSe.PARTES_TAPADAS, MotivoNoSe.SIN_ANIMAL)  # orden alfabético por código
    assert resultado.peso_kg == 170.0  # lo que se alcanzó a estimar, informativo


@pytest.mark.requisito("RF-20")
def test_r22_sin_ningun_peso_el_agregado_no_tiene_peso() -> None:
    resultado = agregar_resultados([foto(None, MotivoNoSe.SIN_ANIMAL)], VERSION)
    assert resultado.peso_kg is None
    assert not resultado.confiable


@pytest.mark.requisito("RF-20")
def test_r20_peso_fuera_de_rango_no_es_confiable() -> None:
    resultado = agregar_resultados([foto(680.0)], VERSION)
    assert not resultado.confiable
    assert resultado.motivos == (MotivoNoSe.FUERA_DE_RANGO,)
    assert resultado.peso_kg == 680.0


@pytest.mark.requisito("RF-20")
def test_r20_usa_el_rango_de_la_version_y_no_uno_fijo() -> None:
    """El mismo peso es válido o no según el rango del config.json que se cargó."""
    angosto = VersionModelo("v-angosta", 100.0, 200.0)
    assert agregar_resultados([foto(250.0)], VERSION).confiable
    assert not agregar_resultados([foto(250.0)], angosto).confiable


@pytest.mark.requisito("RF-20")
def test_r20_los_extremos_del_rango_son_confiables() -> None:
    assert agregar_resultados([foto(36.0)], VERSION).confiable
    assert agregar_resultados([foto(621.0)], VERSION).confiable


@pytest.mark.requisito("RF-20")
def test_r23_el_resultado_registra_la_version_del_modelo() -> None:
    assert agregar_resultados([foto(187.4)], VERSION).version == VERSION


@pytest.mark.requisito("RF-20")
def test_el_peso_agregado_se_redondea_a_un_decimal() -> None:
    assert agregar_resultados([foto(185.94), foto(186.0)], VERSION).peso_kg == 186.0


@pytest.mark.requisito("RF-15")
def test_las_advertencias_pasan_al_resultado() -> None:
    resultado = agregar_resultados([foto(187.4)], VERSION, advertencias=("la imagen 2 no trae metadata EXIF",))
    assert resultado.advertencias == ("la imagen 2 no trae metadata EXIF",)


@pytest.mark.requisito("RF-20")
@pytest.mark.parametrize("cantidad", [0, 5])
def test_r22_acepta_de_una_a_cuatro_fotos(cantidad: int) -> None:
    with pytest.raises(ErrorDeDominio) as excinfo:
        agregar_resultados([foto(180.0)] * cantidad, VERSION)
    assert excinfo.value.codigo is CodigoError.CANTIDAD_IMAGENES_INVALIDA


@pytest.mark.requisito("RF-20")
def test_r22_cuatro_fotos_es_el_maximo_valido() -> None:
    assert agregar_resultados([foto(180.0)] * 4, VERSION).confiable


# --- huella -------------------------------------------------------------------------------------


@pytest.mark.requisito("RF-19")
def test_huella_igual_para_las_mismas_imagenes_y_distinta_si_cambian_o_cambia_el_orden() -> None:
    assert huella_de_imagenes(["a", "b"]) == huella_de_imagenes(["a", "b"])
    assert huella_de_imagenes(["a", "b"]) != huella_de_imagenes(["a", "c"])
    assert huella_de_imagenes(["a", "b"]) != huella_de_imagenes(["b", "a"])


@pytest.mark.requisito("RF-19")
def test_huella_exige_al_menos_una_imagen() -> None:
    with pytest.raises(ValueError):
        huella_de_imagenes([])


# --- Trabajo ------------------------------------------------------------------------------------


def trabajo(imagenes: tuple[str, ...] = ("ref-1",)) -> Trabajo:
    return Trabajo.crear(
        IdTrabajo("trb_01"),
        IdEvento("c3b1"),
        huella_de_imagenes(["hash-1"]),
        imagenes,
        creado=T0,
        metadatos={"raza": "Brangus"},
    )


@pytest.mark.requisito("RF-19")
def test_un_trabajo_nuevo_nace_pendiente() -> None:
    t = trabajo()
    assert t.estado is EstadoTrabajo.PENDIENTE
    assert t.metadatos == {"raza": "Brangus"}
    assert t.resultado is None
    assert t.cola_ms is None
    assert t.total_ms is None


@pytest.mark.requisito("RF-19")
@pytest.mark.parametrize("cantidad", [0, 5])
def test_r18_un_trabajo_lleva_de_una_a_cuatro_imagenes(cantidad: int) -> None:
    with pytest.raises(ErrorDeDominio) as excinfo:
        trabajo(tuple(f"ref-{i}" for i in range(cantidad)))
    assert excinfo.value.codigo is CodigoError.CANTIDAD_IMAGENES_INVALIDA


@pytest.mark.requisito("RF-19")
def test_las_fechas_del_trabajo_llevan_zona_horaria() -> None:
    with pytest.raises(ValueError):
        Trabajo.crear(IdTrabajo("t"), IdEvento("e"), "h", ("r",), creado=datetime(2026, 10, 25, 12, 0, 0))
    with pytest.raises(ValueError):
        trabajo().iniciar(datetime(2026, 10, 25, 12, 0, 1))


@pytest.mark.requisito("RF-19")
def test_el_trabajo_es_inmutable_y_cada_transicion_devuelve_otro() -> None:
    original = trabajo()
    iniciado = original.iniciar(T0 + timedelta(milliseconds=120))
    assert original.estado is EstadoTrabajo.PENDIENTE
    assert iniciado.estado is EstadoTrabajo.PROCESANDO
    with pytest.raises(dataclasses.FrozenInstanceError):
        original.estado = EstadoTrabajo.FALLIDO  # type: ignore[misc]  # se prueba justamente que no se pueda asignar


@pytest.mark.requisito("RF-19")
def test_idempotencia_mismo_pedido_se_reconoce_por_la_huella() -> None:
    t = trabajo()
    assert t.es_mismo_pedido(huella_de_imagenes(["hash-1"]))
    assert not t.es_mismo_pedido(huella_de_imagenes(["hash-2"]))


@pytest.mark.requisito("RF-20")
def test_r21_camino_feliz_pendiente_procesando_completado_con_tiempos() -> None:
    resultado = agregar_resultados([foto(187.4)], VERSION)
    t = trabajo().iniciar(T0 + timedelta(milliseconds=120))
    t = t.completar(resultado, T0 + timedelta(milliseconds=2050), descarga_ms=90, inferencia_ms=1840)
    assert t.estado is EstadoTrabajo.COMPLETADO
    assert t.resultado == resultado
    assert (t.cola_ms, t.total_ms, t.descarga_ms, t.inferencia_ms) == (120, 2050, 90, 1840)


@pytest.mark.requisito("RF-20")
def test_r21_el_no_se_tambien_termina_completado_y_no_fallido() -> None:
    no_se = agregar_resultados([foto(None, MotivoNoSe.SIN_ANIMAL)], VERSION)
    t = trabajo().iniciar(T0).completar(no_se, T0)
    assert t.estado is EstadoTrabajo.COMPLETADO
    assert t.codigo_error is None
    assert t.resultado is not None
    assert not t.resultado.confiable


@pytest.mark.requisito("RF-22")
def test_una_falla_interna_deja_el_trabajo_fallido_con_codigo_y_reintentable() -> None:
    t = trabajo().iniciar(T0).fallar(CodigoError.TIEMPO_EXCEDIDO, T0 + timedelta(seconds=45), reintentable=True)
    assert t.estado is EstadoTrabajo.FALLIDO
    assert t.codigo_error is CodigoError.TIEMPO_EXCEDIDO
    assert t.reintentable is True
    assert t.resultado is None
    assert t.total_ms == 45_000


@pytest.mark.requisito("RF-22")
def test_un_error_de_entrada_no_puede_dejar_un_trabajo_fallido() -> None:
    with pytest.raises(ValueError):
        trabajo().iniciar(T0).fallar(CodigoError.FORMATO_NO_ADMITIDO, T0, reintentable=False)


@pytest.mark.requisito("RF-22")
def test_no_hay_vuelta_atras_en_el_estado() -> None:
    resultado = agregar_resultados([foto(187.4)], VERSION)
    pendiente = trabajo()
    procesando = pendiente.iniciar(T0)
    completado = procesando.completar(resultado, T0)
    fallido = procesando.fallar(CodigoError.ERROR_INTERNO, T0, reintentable=False)

    with pytest.raises(TransicionInvalida):
        pendiente.completar(resultado, T0)  # no se completa sin haber empezado
    with pytest.raises(TransicionInvalida):
        pendiente.fallar(CodigoError.ERROR_INTERNO, T0, reintentable=False)
    with pytest.raises(TransicionInvalida):
        procesando.iniciar(T0)
    for terminal in (completado, fallido):
        with pytest.raises(TransicionInvalida):
            terminal.iniciar(T0)
        with pytest.raises(TransicionInvalida):
            terminal.completar(resultado, T0)
        with pytest.raises(TransicionInvalida):
            terminal.fallar(CodigoError.ERROR_INTERNO, T0, reintentable=False)
