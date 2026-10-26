import pytest

from bv_estimacion.aplicacion.casos_uso.consultar_estimacion import ConsultarEstimacion
from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio
from bv_estimacion.dominio.estimacion import Trabajo
from bv_estimacion.dominio.valores import IdEvento, IdTrabajo, OrigenImagen, ReferenciaImagen
from tests.fakes import T0, RepositorioEnMemoria


@pytest.mark.requisito("RF-19")
async def test_devuelve_el_trabajo_con_su_estado_actual() -> None:
    repo = RepositorioEnMemoria()
    referencia = ReferenciaImagen(OrigenImagen.ALMACEN_TEMPORAL, "tmp-1", "a" * 64)
    trabajo = Trabajo.crear(IdTrabajo("trb_01"), IdEvento("evt-1"), [referencia], T0)
    repo.insertar(trabajo)

    assert await ConsultarEstimacion(repo).ejecutar(IdTrabajo("trb_01")) == trabajo


@pytest.mark.requisito("RF-19")
async def test_un_trabajo_inexistente_o_vencido_es_trabajo_no_encontrado() -> None:
    with pytest.raises(ErrorDeDominio) as error:
        await ConsultarEstimacion(RepositorioEnMemoria()).ejecutar(IdTrabajo("trb_99"))
    assert error.value.codigo is CodigoError.TRABAJO_NO_ENCONTRADO
