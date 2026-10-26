"""Lo que el `lifespan` arma una vez y los routers consumen. Es el único punto donde se juntan los puertos con sus
adaptadores; los tests lo reemplazan pasando otra fábrica del modelo a `create_app`."""

from dataclasses import dataclass

import httpx
from fastapi import Request

from bv_estimacion.aplicacion.casos_uso.consultar_estimacion import ConsultarEstimacion
from bv_estimacion.aplicacion.casos_uso.solicitar_estimacion import SolicitarEstimacion
from bv_estimacion.aplicacion.casos_uso.validar_imagen import ValidarImagen
from bv_estimacion.infraestructura.config import Configuracion
from bv_estimacion.infraestructura.ml.adaptador_modelo import AdaptadorModelo
from bv_estimacion.infraestructura.ml.config_modelo import FichaModelo


@dataclass(frozen=True, slots=True)
class Contenedor:
    configuracion: Configuracion
    ficha: FichaModelo
    adaptador: AdaptadorModelo
    solicitar: SolicitarEstimacion
    consultar: ConsultarEstimacion
    validar: ValidarImagen
    cliente: httpx.AsyncClient


def obtener_contenedor(request: Request) -> Contenedor:
    contenedor: Contenedor = request.app.state.contenedor
    return contenedor
