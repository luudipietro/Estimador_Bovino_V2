import os
import re
import time
from pathlib import Path

import pytest

from bv_estimacion.infraestructura.trabajos.almacen_temporal import AlmacenTemporalEnDisco

TTL_S = 3600


class RelojAdelantable:
    """Hora de pared más un desfase: permite 'esperar' el TTL sin esperar."""

    def __init__(self) -> None:
        self.desfase_s = 0.0

    def __call__(self) -> float:
        return time.time() + self.desfase_s


def armar(carpeta: Path) -> tuple[AlmacenTemporalEnDisco, RelojAdelantable]:
    reloj = RelojAdelantable()
    return AlmacenTemporalEnDisco(carpeta, TTL_S, reloj), reloj


def envejecer(ruta: Path, segundos: float) -> None:
    pasado = time.time() - segundos
    os.utime(ruta, (pasado, pasado))


@pytest.mark.requisito("RF-19")
async def test_guardar_y_obtener_devuelve_el_mismo_contenido(tmp_path: Path) -> None:
    almacen, _ = armar(tmp_path)
    referencia = await almacen.guardar(b"foto-1")
    assert re.fullmatch(r"[0-9a-f]{32}", referencia)
    assert await almacen.obtener(referencia) == b"foto-1"


@pytest.mark.requisito("RF-19")
async def test_cada_guardado_tiene_su_propia_referencia(tmp_path: Path) -> None:
    almacen, _ = armar(tmp_path)
    a, b = await almacen.guardar(b"a"), await almacen.guardar(b"a")
    assert a != b


@pytest.mark.requisito("RF-19")
async def test_crea_la_carpeta_si_no_existe_y_no_deja_archivos_parciales(tmp_path: Path) -> None:
    carpeta = tmp_path / "no" / "existe" / "todavia"
    almacen, _ = armar(carpeta)
    await almacen.guardar(b"foto")
    assert [p.suffix for p in carpeta.iterdir()] == [".img"]


@pytest.mark.requisito("RF-19")
async def test_una_referencia_inexistente_devuelve_none(tmp_path: Path) -> None:
    almacen, _ = armar(tmp_path)
    assert await almacen.obtener("a" * 32) is None


@pytest.mark.requisito("RF-19")
async def test_borrar_elimina_y_es_idempotente(tmp_path: Path) -> None:
    almacen, _ = armar(tmp_path)
    referencia = await almacen.guardar(b"foto")
    await almacen.borrar(referencia)
    await almacen.borrar(referencia)  # segunda vez: no es un error
    await almacen.borrar("b" * 32)  # nunca existió
    assert await almacen.obtener(referencia) is None


REFERENCIAS_INVALIDAS = ["", "../secreto", "..", "a" * 31, "a" * 33, "A" * 32, "g" * 32, "a/b" + "c" * 29]


@pytest.mark.requisito("RF-19")
@pytest.mark.parametrize("referencia", REFERENCIAS_INVALIDAS)
async def test_una_referencia_con_formato_ajeno_se_trata_como_inexistente_y_no_sale_de_la_carpeta(
    tmp_path: Path, referencia: str
) -> None:
    carpeta = tmp_path / "almacen"
    almacen, _ = armar(carpeta)
    await almacen.guardar(b"propio")
    afuera = tmp_path / "secreto.img"
    afuera.write_bytes(b"no me toques")

    assert await almacen.obtener(referencia) is None
    await almacen.borrar(referencia)

    assert afuera.read_bytes() == b"no me toques"


# --- TTL de respaldo ----------------------------------------------------------------------------


@pytest.mark.requisito("RF-22")
async def test_una_imagen_vencida_ya_no_se_entrega_y_se_borra(tmp_path: Path) -> None:
    almacen, reloj = armar(tmp_path)
    referencia = await almacen.guardar(b"foto")

    reloj.desfase_s = TTL_S - 60
    assert await almacen.obtener(referencia) == b"foto"

    reloj.desfase_s = TTL_S + 60
    assert await almacen.obtener(referencia) is None
    assert list(tmp_path.glob("*.img")) == []


@pytest.mark.requisito("RF-22")
async def test_el_barrido_borra_las_huerfanas_vencidas_y_deja_las_vigentes(tmp_path: Path) -> None:
    almacen, _ = armar(tmp_path)
    huerfana = await almacen.guardar(b"quedo de un proceso caido")
    vigente = await almacen.guardar(b"reciente")
    envejecer(tmp_path / f"{huerfana}.img", TTL_S + 10)

    assert await almacen.barrer_vencidos() == 1

    assert await almacen.obtener(huerfana) is None
    assert await almacen.obtener(vigente) == b"reciente"


@pytest.mark.requisito("RF-22")
async def test_el_barrido_limpia_restos_pero_no_toca_archivos_ajenos(tmp_path: Path) -> None:
    almacen, _ = armar(tmp_path)
    resto = tmp_path / ("c" * 32 + ".parcial")
    resto.write_bytes(b"escritura a medias")
    ajeno = tmp_path / "notas.txt"
    ajeno.write_bytes(b"de otra persona")
    envejecer(resto, TTL_S + 10)
    envejecer(ajeno, TTL_S + 10)

    assert await almacen.barrer_vencidos() == 1

    assert not resto.exists()
    assert ajeno.read_bytes() == b"de otra persona"


@pytest.mark.requisito("RF-22")
async def test_barrer_una_carpeta_que_todavia_no_existe_no_falla(tmp_path: Path) -> None:
    almacen, _ = armar(tmp_path / "nada")
    assert await almacen.barrer_vencidos() == 0


@pytest.mark.requisito("RF-22")
async def test_si_otro_hilo_borra_un_archivo_durante_el_barrido_el_barrido_no_falla(tmp_path: Path) -> None:
    """Carrera real: `ProcesarEstimacion` (o `obtener`) borra una imagen entre el listado y la consulta del barrido.
    El reloj inyectado borra los archivos justo antes de que el barrido los consulte."""

    def ahora_que_borra_los_archivos() -> float:
        for ruta in tmp_path.glob("*.img"):
            ruta.unlink(missing_ok=True)
        return time.time()

    almacen = AlmacenTemporalEnDisco(tmp_path, TTL_S, ahora_que_borra_los_archivos)
    await almacen.guardar(b"a")
    await almacen.guardar(b"b")

    assert await almacen.barrer_vencidos() == 0  # no lanza, y no cuenta lo que ya no estaba
    assert list(tmp_path.glob("*.img")) == []
