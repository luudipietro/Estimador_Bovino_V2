from datetime import timedelta

import pytest

from bv_estimacion.infraestructura.reloj import DormidorAnyio, GeneradorIdsUlid, RelojSistema

ALFABETO = set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")


@pytest.mark.requisito("RF-19")
def test_el_reloj_devuelve_la_hora_con_zona_horaria() -> None:
    ahora = RelojSistema().ahora()
    assert ahora.utcoffset() == timedelta(0)


@pytest.mark.requisito("RF-19")
async def test_el_dormidor_espera_de_verdad() -> None:
    import time

    inicio = time.perf_counter()
    await DormidorAnyio().dormir(0.05)
    assert time.perf_counter() - inicio >= 0.04


@pytest.mark.requisito("RF-19")
def test_el_id_tiene_el_formato_trb_mas_ulid_de_26_caracteres() -> None:
    id_trabajo = str(GeneradorIdsUlid().nuevo_id_trabajo())
    assert id_trabajo.startswith("trb_")
    ulid = id_trabajo.removeprefix("trb_")
    assert len(ulid) == 26
    assert set(ulid) <= ALFABETO


@pytest.mark.requisito("RF-19")
def test_los_ids_son_unicos_y_salen_en_orden_creciente_incluso_en_el_mismo_milisegundo() -> None:
    generador = GeneradorIdsUlid()
    ids = [str(generador.nuevo_id_trabajo()) for _ in range(5000)]  # muchos más que milisegundos transcurridos
    assert len(set(ids)) == 5000
    assert ids == sorted(ids)


@pytest.mark.requisito("RF-19")
def test_dos_generadores_distintos_no_repiten_ids() -> None:
    a, b = GeneradorIdsUlid(), GeneradorIdsUlid()
    ids = {str(a.nuevo_id_trabajo()) for _ in range(1000)} | {str(b.nuevo_id_trabajo()) for _ in range(1000)}
    assert len(ids) == 2000


@pytest.mark.requisito("RF-19")
def test_si_el_azar_se_desborda_dentro_del_mismo_milisegundo_el_id_sigue_siendo_creciente() -> None:
    generador = GeneradorIdsUlid()
    anterior = str(generador.nuevo_id_trabajo())
    generador._ultimo_azar = (1 << 80) - 1  # el próximo incremento desborda
    siguiente = str(generador.nuevo_id_trabajo())
    assert siguiente > anterior
    assert len(siguiente) == len("trb_") + 26
