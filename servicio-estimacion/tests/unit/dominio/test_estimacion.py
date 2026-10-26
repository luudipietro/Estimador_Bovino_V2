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
from bv_estimacion.dominio.valores import IdEvento, IdTrabajo, OrigenImagen, ReferenciaImagen, VersionModelo

VERSION = VersionModelo("2026-09-25 / AcmeAI-BMGF", 36.0, 621.0)
T0 = datetime(2026, 10, 25, 12, 0, 0, tzinfo=UTC)


def sha(letra: str) -> str:
    """Un SHA-256 de mentira pero bien formado: 64 caracteres hexadecimales."""
    return letra * 64


def ref(letra: str, origen: OrigenImagen = OrigenImagen.ALMACEN_TEMPORAL) -> ReferenciaImagen:
    return ReferenciaImagen(origen, f"ubicacion-{letra}", sha(letra))


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
@pytest.mark.parametrize(
    "fotos",
    [
        [foto(150.0), foto(160.0)],  # confiables, dentro del rango
        [foto(150.0, MotivoNoSe.IMAGEN_BORROSA), foto(None, MotivoNoSe.SIN_ANIMAL)],  # ninguna confiable
        [foto(700.0), foto(710.0)],  # confiables pero la mediana cae fuera del rango (R-20)
    ],
)
def test_el_agregado_conserva_el_resultado_de_cada_foto_en_el_orden_recibido(fotos: list[ResultadoFoto]) -> None:
    """El contrato devuelve `imagenes[]` con el detalle de cada foto, no solo la mediana."""
    assert agregar_resultados(fotos, VERSION).fotos == tuple(fotos)


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
def test_huella_igual_para_las_mismas_imagenes_y_distinta_si_cambian() -> None:
    assert huella_de_imagenes([sha("a"), sha("b")]) == huella_de_imagenes([sha("a"), sha("b")])
    assert huella_de_imagenes([sha("a"), sha("b")]) != huella_de_imagenes([sha("a"), sha("c")])
    assert huella_de_imagenes([sha("a")]) != huella_de_imagenes([sha("a"), sha("b")])


@pytest.mark.requisito("RF-17")
def test_una_permutacion_de_las_mismas_imagenes_da_la_misma_huella() -> None:
    """Un reintento con otro orden (sincronización offline, RF-17) es el mismo pedido, no un 409."""
    esperada = huella_de_imagenes([sha("a"), sha("b"), sha("c")])
    assert huella_de_imagenes([sha("c"), sha("a"), sha("b")]) == esperada
    assert huella_de_imagenes([sha("b"), sha("c"), sha("a")]) == esperada


@pytest.mark.requisito("RF-17")
def test_la_huella_no_distingue_mayusculas_en_el_hash() -> None:
    assert huella_de_imagenes([sha("a")]) == huella_de_imagenes([sha("A")])


@pytest.mark.requisito("RF-19")
def test_huella_exige_al_menos_una_imagen() -> None:
    with pytest.raises(ValueError):
        huella_de_imagenes([])


# --- Trabajo ------------------------------------------------------------------------------------


def trabajo(imagenes: tuple[ReferenciaImagen, ...] = (ref("a"),)) -> Trabajo:
    return Trabajo.crear(
        IdTrabajo("trb_01"),
        IdEvento("c3b1"),
        imagenes,
        creado=T0,
        metadatos={"raza": "Brangus"},
        url_notificacion="https://backend.example/callback",
    )


@pytest.mark.requisito("RF-19")
def test_un_trabajo_nuevo_nace_pendiente() -> None:
    t = trabajo()
    assert t.estado is EstadoTrabajo.PENDIENTE
    assert t.metadatos == {"raza": "Brangus"}
    assert t.url_notificacion == "https://backend.example/callback"
    assert t.resultado is None
    assert t.cola_ms is None
    assert t.total_ms is None


@pytest.mark.requisito("RF-19")
@pytest.mark.parametrize("cantidad", [0, 5])
def test_r18_un_trabajo_lleva_de_una_a_cuatro_imagenes(cantidad: int) -> None:
    with pytest.raises(ErrorDeDominio) as excinfo:
        trabajo(tuple(ref(letra) for letra in "abcde"[:cantidad]))
    assert excinfo.value.codigo is CodigoError.CANTIDAD_IMAGENES_INVALIDA


@pytest.mark.requisito("RF-19")
def test_las_fechas_del_trabajo_llevan_zona_horaria() -> None:
    with pytest.raises(ValueError):
        Trabajo.crear(IdTrabajo("t"), IdEvento("e"), (ref("a"),), creado=datetime(2026, 10, 25, 12, 0, 0))
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
    assert t.es_mismo_pedido(huella_de_imagenes([sha("a")]))
    assert not t.es_mismo_pedido(huella_de_imagenes([sha("b")]))


@pytest.mark.requisito("RF-17")
def test_el_reintento_con_otro_orden_es_el_mismo_pedido() -> None:
    original = trabajo((ref("a"), ref("b"), ref("c")))
    assert original.es_mismo_pedido(huella_de_imagenes([sha("c"), sha("a"), sha("b")]))


@pytest.mark.requisito("RF-17")
def test_las_imagenes_conservan_el_orden_del_primer_pedido() -> None:
    """Los índices ("imagen 1", "imagen 2") salen del primer pedido, aunque la huella ignore el orden."""
    original = trabajo((ref("c"), ref("a"), ref("b")))
    assert [i.sha256[0] for i in original.imagenes] == ["c", "a", "b"]
    assert original.huella_imagenes == huella_de_imagenes([sha("a"), sha("b"), sha("c")])


@pytest.mark.requisito("RF-19")
def test_el_trabajo_mezcla_origenes_y_guarda_la_ubicacion() -> None:
    t = trabajo((ref("a"), ref("b", OrigenImagen.URL)))
    assert [i.origen for i in t.imagenes] == [OrigenImagen.ALMACEN_TEMPORAL, OrigenImagen.URL]


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
