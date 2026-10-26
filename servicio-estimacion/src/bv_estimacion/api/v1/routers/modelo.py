from fastapi import APIRouter, Depends

from bv_estimacion.api.dependencias import Contenedor, obtener_contenedor
from bv_estimacion.api.seguridad import exigir_token
from bv_estimacion.api.v1.esquemas.modelo import FichaModeloRespuesta

router = APIRouter(prefix="/v1", tags=["modelo"], dependencies=[Depends(exigir_token)])


@router.get("/modelo", response_model=FichaModeloRespuesta, summary="Versión y rango de validez del modelo")
async def modelo(contenedor: Contenedor = Depends(obtener_contenedor)) -> FichaModeloRespuesta:  # noqa: B008
    ficha = contenedor.ficha
    return FichaModeloRespuesta(
        version=ficha.version.version,
        rango_validez_kg=ficha.version.rango_validez_kg,
        mape_test=ficha.mape_test,
        ic95_test=ficha.ic95_test,
        fecha_entrenamiento=ficha.fecha_entrenamiento,
    )
