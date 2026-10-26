import hashlib
from dataclasses import dataclass

import pytest

from bv_estimacion.aplicacion.casos_uso.solicitar_estimacion import SolicitarEstimacion
from bv_estimacion.aplicacion.dto import ReferenciaExterna, SolicitarEstimacionCmd
from bv_estimacion.aplicacion.puertos import ColaLlena
from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio
from bv_estimacion.dominio.estimacion import EstadoTrabajo, Trabajo
from bv_estimacion.dominio.valores import IdEvento, IdTrabajo, OrigenImagen, ReferenciaImagen
from tests.fakes import (
    T0,
    AlmacenTemporalEnMemoria,
    Bitacora,
    ColaEnMemoria,
    GeneradorIdsSecuencial,
    RelojFijo,
    RepositorioEnMemoria,
)


def sha(contenido: bytes) -> str:
    return hashlib.sha256(contenido).hexdigest()


@dataclass
class Contexto:
    caso: SolicitarEstimacion
    repo: RepositorioEnMemoria
    cola: ColaEnMemoria
    almacen: AlmacenTemporalEnMemoria
    bitacora: Bitacora


def armar(capacidad: int = 10) -> Contexto:
    bitacora = Bitacora()
    repo = RepositorioEnMemoria(bitacora)
    cola = ColaEnMemoria(capacidad, bitacora)
    almacen = AlmacenTemporalEnMemoria(bitacora)
    caso = SolicitarEstimacion(repo, cola, almacen, GeneradorIdsSecuencial(), RelojFijo())
    return Contexto(caso, repo, cola, almacen, bitacora)


def pedido(*imagenes: bytes, id_evento: str = "evt-1", **kwargs: object) -> SolicitarEstimacionCmd:
    return SolicitarEstimacionCmd(IdEvento(id_evento), imagenes=imagenes, **kwargs)  # type: ignore[arg-type]


# --- Camino feliz -------------------------------------------------------------------------------


@pytest.mark.requisito("RF-19")
async def test_un_pedido_nuevo_queda_pendiente_y_en_la_cola() -> None:
    ctx = armar()
    aceptada = await ctx.caso.ejecutar(pedido(b"foto-1"))

    assert aceptada.nuevo
    assert aceptada.trabajo.estado is EstadoTrabajo.PENDIENTE
    assert aceptada.trabajo.id == IdTrabajo("trb_01")  # lo generó el GeneradorIds
    assert aceptada.trabajo.creado == T0  # lo dio el Reloj
    assert ctx.cola.ids == [aceptada.trabajo.id]
    assert ctx.repo.trabajos[aceptada.trabajo.id] == aceptada.trabajo


@pytest.mark.requisito("RF-19")
async def test_las_imagenes_se_guardan_antes_de_responder_y_el_trabajo_lleva_su_sha256() -> None:
    ctx = armar()
    aceptada = await ctx.caso.ejecutar(pedido(b"foto-1", b"foto-2"))

    assert sorted(ctx.almacen.objetos.values()) == [b"foto-1", b"foto-2"]
    refs = aceptada.trabajo.imagenes
    assert [r.origen for r in refs] == [OrigenImagen.ALMACEN_TEMPORAL] * 2
    assert [r.sha256 for r in refs] == [sha(b"foto-1"), sha(b"foto-2")]  # en el orden recibido
    assert all(ctx.almacen.objetos[r.ubicacion] in (b"foto-1", b"foto-2") for r in refs)


@pytest.mark.requisito("RF-19")
async def test_el_orden_es_chequear_cola_guardar_imagen_guardar_trabajo_y_encolar_al_final() -> None:
    ctx = armar()
    await ctx.caso.ejecutar(pedido(b"foto-1"))
    assert ctx.bitacora.eventos == ["cola.hay_lugar", "almacen.guardar", "repo.agregar", "cola.encolar"]


@pytest.mark.requisito("RF-19")
async def test_metadatos_y_url_de_notificacion_se_guardan_en_el_trabajo() -> None:
    ctx = armar()
    aceptada = await ctx.caso.ejecutar(
        pedido(b"foto-1", metadatos={"raza": "Brangus"}, url_notificacion="https://backend.example/cb"),
    )
    assert aceptada.trabajo.metadatos == {"raza": "Brangus"}  # R-26: se devuelven sin modificar
    assert aceptada.trabajo.url_notificacion == "https://backend.example/cb"


@pytest.mark.requisito("RF-19")
async def test_con_referencias_externas_no_se_toca_el_almacen_temporal() -> None:
    ctx = armar()
    cmd = SolicitarEstimacionCmd(
        IdEvento("evt-1"),
        referencias=(ReferenciaExterna("https://almacen.example/a.jpg", sha(b"a")),),
    )
    aceptada = await ctx.caso.ejecutar(cmd)

    assert ctx.almacen.objetos == {}
    assert "almacen.guardar" not in ctx.bitacora.eventos
    [referencia] = aceptada.trabajo.imagenes
    assert referencia.origen is OrigenImagen.URL
    assert referencia.ubicacion == "https://almacen.example/a.jpg"
    assert referencia.sha256 == sha(b"a")


@pytest.mark.requisito("RF-19")
async def test_una_referencia_con_sha256_mal_formado_no_pasa() -> None:
    ctx = armar()
    cmd = SolicitarEstimacionCmd(IdEvento("evt-1"), referencias=(ReferenciaExterna("https://x.example/a", "xyz"),))
    with pytest.raises(ValueError):
        await ctx.caso.ejecutar(cmd)
    assert ctx.repo.trabajos == {}


# --- Cantidad de imágenes -----------------------------------------------------------------------


@pytest.mark.requisito("RF-19")
async def test_r18_imagenes_y_referencias_a_la_vez_es_invalido() -> None:
    ctx = armar()
    cmd = SolicitarEstimacionCmd(
        IdEvento("evt-1"),
        imagenes=(b"foto-1",),
        referencias=(ReferenciaExterna("https://x.example/a", sha(b"a")),),
    )
    with pytest.raises(ErrorDeDominio) as error:
        await ctx.caso.ejecutar(cmd)
    assert error.value.codigo is CodigoError.CANTIDAD_IMAGENES_INVALIDA
    assert ctx.almacen.objetos == {}


@pytest.mark.requisito("RF-19")
@pytest.mark.parametrize("cantidad", [0, 5])
async def test_r18_se_aceptan_de_una_a_cuatro_imagenes(cantidad: int) -> None:
    ctx = armar()
    with pytest.raises(ErrorDeDominio) as error:
        await ctx.caso.ejecutar(pedido(*[f"foto-{i}".encode() for i in range(cantidad)]))
    assert error.value.codigo is CodigoError.CANTIDAD_IMAGENES_INVALIDA
    assert ctx.bitacora.eventos == []  # se rechazó antes de tocar nada


# --- Idempotencia -------------------------------------------------------------------------------


@pytest.mark.requisito("RF-19")
async def test_el_mismo_pedido_devuelve_el_mismo_trabajo_sin_duplicar_nada() -> None:
    ctx = armar()
    primero = await ctx.caso.ejecutar(pedido(b"foto-1"))
    segundo = await ctx.caso.ejecutar(pedido(b"foto-1"))

    assert not segundo.nuevo
    assert segundo.trabajo.id == primero.trabajo.id
    assert len(ctx.repo.trabajos) == 1
    assert len(ctx.almacen.objetos) == 1  # no se guardó la imagen otra vez
    assert len(ctx.cola.ids) == 1  # no se encoló otra vez


@pytest.mark.requisito("RF-17")
async def test_un_reintento_con_las_imagenes_en_otro_orden_devuelve_el_trabajo_original() -> None:
    """Sincronización offline (RF-17): el reintento puede reordenar las fotos y no debe dar 409."""
    ctx = armar()
    primero = await ctx.caso.ejecutar(pedido(b"foto-1", b"foto-2"))
    reintento = await ctx.caso.ejecutar(pedido(b"foto-2", b"foto-1"))

    assert not reintento.nuevo
    assert reintento.trabajo.id == primero.trabajo.id
    # Los índices siguen el orden del primer pedido
    assert [r.sha256 for r in reintento.trabajo.imagenes] == [sha(b"foto-1"), sha(b"foto-2")]
    assert len(ctx.almacen.objetos) == 2


@pytest.mark.requisito("RF-19")
async def test_el_mismo_id_evento_con_otra_imagen_es_un_conflicto_y_no_toca_el_original() -> None:
    ctx = armar()
    original = await ctx.caso.ejecutar(pedido(b"foto-1"))

    with pytest.raises(ErrorDeDominio) as error:
        await ctx.caso.ejecutar(pedido(b"otra-foto"))

    assert error.value.codigo is CodigoError.ID_EVENTO_EN_CONFLICTO
    assert ctx.repo.trabajos == {original.trabajo.id: original.trabajo}
    assert list(ctx.almacen.objetos.values()) == [b"foto-1"]


@pytest.mark.requisito("RF-19")
async def test_la_idempotencia_con_urls_no_depende_de_la_firma_de_la_url() -> None:
    ctx = armar()

    def con_url(url: str) -> SolicitarEstimacionCmd:
        return SolicitarEstimacionCmd(IdEvento("evt-1"), referencias=(ReferenciaExterna(url, sha(b"a")),))

    primero = await ctx.caso.ejecutar(con_url("https://almacen.example/a.jpg?firma=AAA"))
    reintento = await ctx.caso.ejecutar(con_url("https://almacen.example/a.jpg?firma=BBB"))
    assert not reintento.nuevo
    assert reintento.trabajo.id == primero.trabajo.id


# --- Cola llena: rechazar antes de persistir ----------------------------------------------------


@pytest.mark.requisito("RF-19")
async def test_con_la_cola_llena_se_rechaza_antes_de_guardar_el_trabajo_o_la_imagen() -> None:
    ctx = armar(capacidad=0)
    with pytest.raises(ErrorDeDominio) as error:
        await ctx.caso.ejecutar(pedido(b"foto-1"))

    assert error.value.codigo is CodigoError.COLA_LLENA
    assert ctx.repo.trabajos == {}
    assert ctx.almacen.objetos == {}
    assert ctx.bitacora.eventos == ["cola.hay_lugar"]  # no se persistió nada


@pytest.mark.requisito("RF-19")
async def test_si_la_cola_se_llena_justo_al_encolar_se_revierte_todo() -> None:
    """Otro pedido ocupó el último lugar entre `hay_lugar` y `encolar`: no quedan huérfanos."""
    ctx = armar()
    ctx.cola.error_al_encolar = ColaLlena()

    with pytest.raises(ErrorDeDominio) as error:
        await ctx.caso.ejecutar(pedido(b"foto-1", b"foto-2"))

    assert error.value.codigo is CodigoError.COLA_LLENA
    assert ctx.repo.trabajos == {}
    assert ctx.almacen.objetos == {}
    assert ctx.cola.ids == []


@pytest.mark.requisito("RF-19")
async def test_si_encolar_falla_por_otra_causa_se_revierte_y_se_propaga_el_error_original() -> None:
    ctx = armar()
    ctx.cola.error_al_encolar = RuntimeError("la cola se cayó")

    with pytest.raises(RuntimeError, match="la cola se cayó"):
        await ctx.caso.ejecutar(pedido(b"foto-1"))

    assert ctx.repo.trabajos == {}
    assert ctx.almacen.objetos == {}


@pytest.mark.requisito("RF-19")
async def test_si_falla_el_guardado_de_una_imagen_se_borran_las_que_ya_se_guardaron() -> None:
    ctx = armar()
    ctx.almacen.fallar_en_el_guardado_numero = 2

    with pytest.raises(OSError, match="disco lleno"):
        await ctx.caso.ejecutar(pedido(b"foto-1", b"foto-2"))

    assert ctx.almacen.objetos == {}
    assert ctx.repo.trabajos == {}
    assert ctx.cola.ids == []


@pytest.mark.requisito("RF-19")
async def test_si_falla_la_reversion_no_se_tapa_el_error_original() -> None:
    """La limpieza es de mejor esfuerzo: si borrar o eliminar también fallan, el llamador igual ve la causa real."""
    ctx = armar()
    ctx.cola.error_al_encolar = RuntimeError("la cola se cayó")
    ctx.repo.error_al_eliminar = OSError("no se pudo eliminar")
    ctx.almacen.error_al_borrar = OSError("no se pudo borrar")

    with pytest.raises(RuntimeError, match="la cola se cayó"):
        await ctx.caso.ejecutar(pedido(b"foto-1"))


# --- Carreras por `id_evento` -------------------------------------------------------------------


def trabajo_ajeno(contenido: bytes) -> Trabajo:
    """Un trabajo con el mismo `id_evento`, creado por otro pedido que ganó la carrera."""
    referencia = ReferenciaImagen(OrigenImagen.ALMACEN_TEMPORAL, "tmp-ajeno", sha(contenido))
    return Trabajo.crear(IdTrabajo("trb_ajeno"), IdEvento("evt-1"), [referencia], T0)


@pytest.mark.requisito("RF-19")
async def test_si_otro_pedido_igual_gana_la_carrera_se_devuelve_el_suyo_y_se_limpia_lo_propio() -> None:
    ctx = armar()
    ctx.repo.antes_de_agregar = lambda: ctx.repo.insertar(trabajo_ajeno(b"foto-1"))

    aceptada = await ctx.caso.ejecutar(pedido(b"foto-1"))

    assert not aceptada.nuevo
    assert aceptada.trabajo.id == IdTrabajo("trb_ajeno")
    assert ctx.almacen.objetos == {}  # la imagen que guardamos de más se borró
    assert ctx.cola.ids == []  # y no se encoló dos veces


@pytest.mark.requisito("RF-19")
async def test_si_otro_pedido_distinto_gana_la_carrera_es_un_conflicto_y_se_limpia_lo_propio() -> None:
    ctx = armar()
    ctx.repo.antes_de_agregar = lambda: ctx.repo.insertar(trabajo_ajeno(b"foto-ajena"))

    with pytest.raises(ErrorDeDominio) as error:
        await ctx.caso.ejecutar(pedido(b"foto-1"))

    assert error.value.codigo is CodigoError.ID_EVENTO_EN_CONFLICTO
    assert ctx.almacen.objetos == {}
    assert list(ctx.repo.trabajos) == [IdTrabajo("trb_ajeno")]
