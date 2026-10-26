from fastapi import APIRouter, Request, Response

from bv_estimacion.api.dependencias import Contenedor
from bv_estimacion.api.v1.esquemas.modelo import Salud

router = APIRouter(tags=["salud"])

_NO_CARGADO = Salud(
    estado="no_disponible",
    modelo_cargado=False,
    version_modelo=None,
    instancias=0,
    error="El modelo no está cargado.",
)


@router.get(
    "/salud",
    response_model=Salud,
    responses={503: {"model": Salud, "description": "El modelo no está disponible"}},
    summary="Liveness y readiness",
)
async def salud(request: Request, respuesta: Response) -> Salud:
    """La única ruta sin autenticación. Con el arranque normal el modelo siempre está cargado (la app no arranca sin
    él), pero si no lo estuviera esto lo dice en vez de mentir."""
    respuesta.headers["Cache-Control"] = "no-store"
    contenedor: Contenedor | None = getattr(request.app.state, "contenedor", None)
    if contenedor is None or not contenedor.adaptador.cargado:
        respuesta.status_code = 503
        return _NO_CARGADO
    return Salud(
        estado="ok",
        modelo_cargado=True,
        version_modelo=contenedor.ficha.version.version,
        instancias=contenedor.adaptador.instancias,
        error=None,
    )
