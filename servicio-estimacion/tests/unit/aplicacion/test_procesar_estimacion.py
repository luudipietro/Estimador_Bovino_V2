import hashlib
import logging
from dataclasses import dataclass

import anyio
import pytest

from bv_estimacion.aplicacion.casos_uso.procesar_estimacion import ProcesarEstimacion
from bv_estimacion.aplicacion.dto import EstimacionFoto
from bv_estimacion.aplicacion.puertos import ModeloNoDisponible
from bv_estimacion.dominio.errores import CodigoError, MotivoNoSe
from bv_estimacion.dominio.estimacion import EstadoTrabajo, ResultadoFoto, Trabajo
from bv_estimacion.dominio.valores import IdEvento, IdTrabajo, OrigenImagen, ReferenciaImagen
from tests.fakes import (
    T0,
    AlmacenTemporalEnMemoria,
    ModeloConPoolDeUnSlot,
    ModeloFalso,
    NotificadorEspia,
    ObtenedorImagenFalso,
    RelojFijo,
    RepositorioEnMemoria,
    foto_ok,
)

PRESUPUESTO_S = 5.0


def sha(contenido: bytes) -> str:
    return hashlib.sha256(contenido).hexdigest()


@dataclass
class Contexto:
    caso: ProcesarEstimacion
    repo: RepositorioEnMemoria
    almacen: AlmacenTemporalEnMemoria
    obtenedor: ObtenedorImagenFalso
    modelo: ModeloFalso
    notificador: NotificadorEspia
    reloj: RelojFijo


def armar(presupuesto_s: float = PRESUPUESTO_S) -> Contexto:
    reloj = RelojFijo()
    repo = RepositorioEnMemoria()
    almacen = AlmacenTemporalEnMemoria()
    obtenedor = ObtenedorImagenFalso()
    modelo = ModeloFalso(reloj)
    notificador = NotificadorEspia(repo)
    caso = ProcesarEstimacion(modelo, repo, almacen, obtenedor, notificador, reloj, presupuesto_s)
    return Contexto(caso, repo, almacen, obtenedor, modelo, notificador, reloj)


async def preparar(ctx: Contexto, *contenidos: bytes, id_trabajo: str = "trb_01") -> IdTrabajo:
    """Deja en el repo un trabajo pendiente cuyas imágenes están en el almacén temporal."""
    referencias = []
    for contenido in contenidos:
        ubicacion = await ctx.almacen.guardar(contenido)
        referencias.append(ReferenciaImagen(OrigenImagen.ALMACEN_TEMPORAL, ubicacion, sha(contenido)))
    trabajo = Trabajo.crear(IdTrabajo(id_trabajo), IdEvento(f"evt-{id_trabajo}"), referencias, T0)
    ctx.repo.insertar(trabajo)
    return trabajo.id


async def procesar(ctx: Contexto, *contenidos: bytes) -> Trabajo:
    id_trabajo = await preparar(ctx, *contenidos)
    await ctx.caso.ejecutar(id_trabajo)
    return ctx.repo.trabajos[id_trabajo]


# --- Resultado ----------------------------------------------------------------------------------


@pytest.mark.requisito("RF-20")
async def test_camino_feliz_el_trabajo_queda_completado_con_el_peso_y_la_version() -> None:
    ctx = armar()
    ctx.modelo.respuestas[b"foto-1"] = foto_ok(187.4)

    trabajo = await procesar(ctx, b"foto-1")

    assert trabajo.estado is EstadoTrabajo.COMPLETADO
    assert trabajo.resultado is not None
    assert trabajo.resultado.peso_kg == 187.4
    assert trabajo.resultado.confiable
    assert trabajo.resultado.version == ctx.modelo.version  # R-23


@pytest.mark.requisito("RF-20")
async def test_r21_el_no_se_termina_completado_con_su_motivo_y_no_fallido() -> None:
    ctx = armar()
    ctx.modelo.por_defecto = EstimacionFoto(ResultadoFoto(None, False, (MotivoNoSe.SIN_ANIMAL,)))

    trabajo = await procesar(ctx, b"foto-1")

    assert trabajo.estado is EstadoTrabajo.COMPLETADO
    assert trabajo.codigo_error is None
    assert trabajo.resultado is not None
    assert not trabajo.resultado.confiable
    assert trabajo.resultado.motivos == (MotivoNoSe.SIN_ANIMAL,)


@pytest.mark.requisito("RF-20")
async def test_r22_varias_fotos_se_agregan_con_la_mediana_de_las_confiables() -> None:
    ctx = armar()
    ctx.modelo.respuestas = {b"a": foto_ok(180.0), b"b": foto_ok(190.0), b"c": foto_ok(400.0)}

    trabajo = await procesar(ctx, b"a", b"b", b"c")

    assert trabajo.resultado is not None
    assert trabajo.resultado.peso_kg == 190.0
    assert ctx.modelo.llamadas == [b"a", b"b", b"c"]  # en el orden del primer pedido


@pytest.mark.requisito("RF-20")
async def test_r20_un_peso_fuera_del_rango_de_la_version_no_es_confiable() -> None:
    ctx = armar()
    ctx.modelo.por_defecto = foto_ok(680.0)

    trabajo = await procesar(ctx, b"foto-1")

    assert trabajo.estado is EstadoTrabajo.COMPLETADO
    assert trabajo.resultado is not None
    assert not trabajo.resultado.confiable
    assert trabajo.resultado.motivos == (MotivoNoSe.FUERA_DE_RANGO,)


@pytest.mark.requisito("RF-15")
async def test_las_advertencias_de_cada_foto_llevan_su_numero_de_imagen() -> None:
    ctx = armar()
    ctx.modelo.respuestas = {b"a": foto_ok(180.0), b"b": foto_ok(190.0, "no trae metadata EXIF")}

    trabajo = await procesar(ctx, b"a", b"b")

    assert trabajo.resultado is not None
    assert trabajo.resultado.advertencias == ("la imagen 2 no trae metadata EXIF",)


@pytest.mark.requisito("RNF-01")
async def test_se_miden_la_cola_la_descarga_y_la_inferencia() -> None:
    ctx = armar()
    ctx.modelo.avanza_reloj_s = 1.84
    id_trabajo = await preparar(ctx, b"foto-1")
    ctx.reloj.avanzar(segundos=0.12)  # el trabajo esperó en la cola

    await ctx.caso.ejecutar(id_trabajo)

    trabajo = ctx.repo.trabajos[id_trabajo]
    assert trabajo.cola_ms == 120
    assert trabajo.inferencia_ms == 1840
    assert trabajo.total_ms == 1960


# --- Fallas -------------------------------------------------------------------------------------


@pytest.mark.requisito("RF-22")
async def test_una_imagen_que_ya_no_esta_en_el_almacen_es_referencia_inaccesible_y_no_error_interno() -> None:
    ctx = armar()
    id_trabajo = await preparar(ctx, b"foto-1")
    ctx.almacen.objetos.clear()  # venció su TTL

    await ctx.caso.ejecutar(id_trabajo)

    trabajo = ctx.repo.trabajos[id_trabajo]
    assert trabajo.estado is EstadoTrabajo.FALLIDO
    assert trabajo.codigo_error is CodigoError.REFERENCIA_INACCESIBLE
    assert trabajo.reintentable is True
    assert ctx.modelo.llamadas == []


@pytest.mark.requisito("RF-22")
async def test_una_url_que_no_se_puede_descargar_es_referencia_inaccesible() -> None:
    ctx = armar()
    referencia = ReferenciaImagen(OrigenImagen.URL, "https://almacen.example/a.jpg", sha(b"a"))
    ctx.repo.insertar(Trabajo.crear(IdTrabajo("trb_01"), IdEvento("evt-1"), [referencia], T0))

    await ctx.caso.ejecutar(IdTrabajo("trb_01"))

    trabajo = ctx.repo.trabajos[IdTrabajo("trb_01")]
    assert trabajo.estado is EstadoTrabajo.FALLIDO
    assert trabajo.codigo_error is CodigoError.REFERENCIA_INACCESIBLE


@pytest.mark.requisito("RF-22")
async def test_una_url_descargada_se_verifica_contra_el_sha256_declarado() -> None:
    ctx = armar()
    referencia = ReferenciaImagen(OrigenImagen.URL, "https://almacen.example/a.jpg", sha(b"declarada"))
    ctx.repo.insertar(Trabajo.crear(IdTrabajo("trb_01"), IdEvento("evt-1"), [referencia], T0))
    ctx.obtenedor.contenidos["https://almacen.example/a.jpg"] = b"OTRA imagen"  # la clave del objeto se reutilizó

    await ctx.caso.ejecutar(IdTrabajo("trb_01"))

    trabajo = ctx.repo.trabajos[IdTrabajo("trb_01")]
    assert trabajo.codigo_error is CodigoError.REFERENCIA_INACCESIBLE
    assert ctx.modelo.llamadas == []  # nunca se estimó una imagen que no era la declarada


@pytest.mark.requisito("RF-20")
async def test_una_url_valida_se_descarga_y_se_estima() -> None:
    ctx = armar()
    referencia = ReferenciaImagen(OrigenImagen.URL, "https://almacen.example/a.jpg", sha(b"a"))
    ctx.repo.insertar(Trabajo.crear(IdTrabajo("trb_01"), IdEvento("evt-1"), [referencia], T0))
    ctx.obtenedor.contenidos["https://almacen.example/a.jpg"] = b"a"

    await ctx.caso.ejecutar(IdTrabajo("trb_01"))

    assert ctx.repo.trabajos[IdTrabajo("trb_01")].estado is EstadoTrabajo.COMPLETADO


@pytest.mark.requisito("RF-22")
async def test_modelo_no_disponible_deja_el_trabajo_fallido_y_reintentable() -> None:
    ctx = armar()
    ctx.modelo.error = ModeloNoDisponible()

    trabajo = await procesar(ctx, b"foto-1")

    assert trabajo.estado is EstadoTrabajo.FALLIDO
    assert trabajo.codigo_error is CodigoError.MODELO_NO_DISPONIBLE
    assert trabajo.reintentable is True


@pytest.mark.requisito("RF-22")
async def test_un_error_inesperado_es_error_interno_no_reintentable_y_sin_detalles(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ctx = armar()
    ctx.modelo.error = ZeroDivisionError("detalle interno")

    with caplog.at_level(logging.ERROR):
        trabajo = await procesar(ctx, b"foto-1")

    assert trabajo.codigo_error is CodigoError.ERROR_INTERNO
    assert trabajo.reintentable is False
    assert "detalle interno" not in repr(trabajo)  # el estado del trabajo no filtra la causa
    assert any("detalle interno" in (r.exc_text or "") for r in caplog.records)  # va al log


@pytest.mark.requisito("RNF-01")
async def test_si_se_pasa_del_presupuesto_queda_fallido_por_tiempo_excedido_y_reintentable() -> None:
    ctx = armar(presupuesto_s=0.05)
    ctx.modelo.espera_real_s = 1.0

    trabajo = await procesar(ctx, b"foto-1")

    assert trabajo.estado is EstadoTrabajo.FALLIDO
    assert trabajo.codigo_error is CodigoError.TIEMPO_EXCEDIDO
    assert trabajo.reintentable is True


# --- Idempotencia del worker --------------------------------------------------------------------


@pytest.mark.requisito("RF-19")
async def test_un_trabajo_inexistente_no_hace_nada() -> None:
    ctx = armar()
    await ctx.caso.ejecutar(IdTrabajo("trb_99"))
    assert ctx.repo.trabajos == {}
    assert ctx.notificador.notificados == []


@pytest.mark.requisito("RF-19")
async def test_un_trabajo_ya_procesado_no_se_procesa_dos_veces() -> None:
    ctx = armar()
    id_trabajo = await preparar(ctx, b"foto-1")
    await ctx.caso.ejecutar(id_trabajo)
    await ctx.caso.ejecutar(id_trabajo)  # la cola entregó el mensaje repetido

    assert ctx.modelo.llamadas == [b"foto-1"]
    assert len(ctx.notificador.notificados) == 1


# --- Imágenes temporales ------------------------------------------------------------------------


@pytest.mark.requisito("RF-22")
async def test_las_imagenes_temporales_se_borran_al_completar() -> None:
    ctx = armar()
    await procesar(ctx, b"a", b"b")
    assert ctx.almacen.objetos == {}


@pytest.mark.requisito("RF-22")
async def test_las_imagenes_temporales_se_borran_tambien_si_el_trabajo_falla() -> None:
    ctx = armar()
    ctx.modelo.error = ModeloNoDisponible()
    await procesar(ctx, b"a", b"b")
    assert ctx.almacen.objetos == {}


@pytest.mark.requisito("RF-22")
async def test_si_no_se_puede_borrar_una_imagen_el_trabajo_igual_queda_completado() -> None:
    """El TTL de respaldo del almacén la limpia; un fallo al borrar no puede perder el resultado."""
    ctx = armar()
    id_trabajo = await preparar(ctx, b"foto-1")
    ctx.almacen.error_al_borrar = OSError("no se pudo borrar")

    await ctx.caso.ejecutar(id_trabajo)

    assert ctx.repo.trabajos[id_trabajo].estado is EstadoTrabajo.COMPLETADO
    assert len(ctx.notificador.notificados) == 1


@pytest.mark.requisito("RNF-01")
async def test_las_imagenes_temporales_se_borran_tambien_si_se_pasa_del_tiempo() -> None:
    ctx = armar(presupuesto_s=0.05)
    ctx.modelo.espera_real_s = 1.0
    await procesar(ctx, b"a")
    assert ctx.almacen.objetos == {}


@pytest.mark.requisito("RF-22")
async def test_las_imagenes_temporales_se_borran_aunque_cancelen_al_worker() -> None:
    ctx = armar(presupuesto_s=30.0)
    ctx.modelo.espera_real_s = 1.0
    id_trabajo = await preparar(ctx, b"a")

    with anyio.move_on_after(0.05):  # el servicio se apaga a mitad de la inferencia
        await ctx.caso.ejecutar(id_trabajo)

    assert ctx.almacen.objetos == {}


# --- Aviso al backend ---------------------------------------------------------------------------


@pytest.mark.requisito("RF-21")
async def test_se_notifica_despues_de_guardar_el_estado_final() -> None:
    ctx = armar()
    await procesar(ctx, b"foto-1")

    assert [t.estado for t in ctx.notificador.notificados] == [EstadoTrabajo.COMPLETADO]
    assert ctx.notificador.estado_guardado_al_notificar == ["completado"]


@pytest.mark.requisito("RF-21")
async def test_tambien_se_notifica_un_trabajo_fallido() -> None:
    ctx = armar()
    ctx.modelo.error = ModeloNoDisponible()
    await procesar(ctx, b"foto-1")
    assert [t.estado for t in ctx.notificador.notificados] == [EstadoTrabajo.FALLIDO]


@pytest.mark.requisito("RF-21")
async def test_si_no_se_puede_agendar_la_notificacion_el_resultado_no_se_pierde() -> None:
    ctx = armar()
    ctx.notificador.error = RuntimeError("no hay cola de avisos")

    trabajo = await procesar(ctx, b"foto-1")

    assert trabajo.estado is EstadoTrabajo.COMPLETADO
    assert ctx.repo.trabajos[trabajo.id].resultado is not None


@pytest.mark.requisito("RNF-01")
async def test_el_aviso_no_cuenta_dentro_del_presupuesto_de_tiempo() -> None:
    ctx = armar(presupuesto_s=0.1)
    ctx.notificador.espera_real_s = 0.3  # mucho más que el presupuesto

    trabajo = await procesar(ctx, b"foto-1")

    assert trabajo.estado is EstadoTrabajo.COMPLETADO  # el aviso lento no lo hizo vencer
    assert len(ctx.notificador.notificados) == 1


# --- Contrato del hilo del modelo ---------------------------------------------------------------


@pytest.mark.requisito("RNF-01")
async def test_el_timeout_no_libera_el_lugar_del_modelo_y_no_se_acumulan_hilos_colgados() -> None:
    """Python no puede matar el hilo de inferencia: tras un `tiempo_excedido` sigue ocupando su lugar hasta que
    termina de verdad. Los trabajos siguientes esperan ese lugar en vez de arrancar otro hilo."""
    reloj = RelojFijo()
    repo = RepositorioEnMemoria()
    almacen = AlmacenTemporalEnMemoria()
    modelo = ModeloConPoolDeUnSlot()
    caso = ProcesarEstimacion(modelo, repo, almacen, ObtenedorImagenFalso(), NotificadorEspia(repo), reloj, 0.05)
    contexto = Contexto(caso, repo, almacen, ObtenedorImagenFalso(), ModeloFalso(), NotificadorEspia(repo), reloj)

    primero = await preparar(contexto, b"a", id_trabajo="trb_01")
    segundo = await preparar(contexto, b"b", id_trabajo="trb_02")

    await caso.ejecutar(primero)  # se pasa de tiempo: su hilo queda colgado
    assert repo.trabajos[primero].codigo_error is CodigoError.TIEMPO_EXCEDIDO
    assert modelo.hilos_activos == 1

    await caso.ejecutar(segundo)  # pide lugar, no lo hay: espera y vence, sin arrancar otro hilo
    assert repo.trabajos[segundo].codigo_error is CodigoError.TIEMPO_EXCEDIDO
    assert modelo.hilos_activos == 1
    assert modelo.maximo_de_hilos_activos == 1

    modelo.terminar.set()  # el hilo colgado termina de verdad
    await anyio.sleep(0.01)
    assert modelo.hilos_activos == 0

    tercero = await preparar(contexto, b"c", id_trabajo="trb_03")
    await caso.ejecutar(tercero)  # ahora el lugar está libre
    assert repo.trabajos[tercero].estado is EstadoTrabajo.COMPLETADO
    assert modelo.maximo_de_hilos_activos == 1
