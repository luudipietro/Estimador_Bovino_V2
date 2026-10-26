"""Arma la app: `create_app` y el `lifespan`. Sin estado global: todo cuelga de `app.state`.

Arranque (opción A): el modelo se carga **antes de aceptar tráfico**. Si falla la verificación de los pesos o la carga,
la app no arranca: un servicio que acepta pedidos sin modelo solo los fallaría después. Los workers se lanzan recién
cuando el modelo está cargado.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
import httpx
from anyio import to_thread
from fastapi import FastAPI

from bv_estimacion.api.dependencias import Contenedor
from bv_estimacion.api.v1.esquemas.trabajo import serializar_trabajo
from bv_estimacion.aplicacion.casos_uso.consultar_estimacion import ConsultarEstimacion
from bv_estimacion.aplicacion.casos_uso.notificar_resultado import NotificarResultado
from bv_estimacion.aplicacion.casos_uso.procesar_estimacion import ProcesarEstimacion
from bv_estimacion.aplicacion.casos_uso.solicitar_estimacion import SolicitarEstimacion
from bv_estimacion.aplicacion.casos_uso.validar_imagen import ValidarImagen
from bv_estimacion.infraestructura.config import Configuracion
from bv_estimacion.infraestructura.imagenes.descarga_http import ObtenedorImagenHttp
from bv_estimacion.infraestructura.ml.adaptador_modelo import AdaptadorModelo
from bv_estimacion.infraestructura.ml.config_modelo import cargar_ficha_modelo
from bv_estimacion.infraestructura.ml.fabrica import ModeloConfigurado, fabrica_real
from bv_estimacion.infraestructura.ml.verificacion import verificar_hashes_modelos
from bv_estimacion.infraestructura.notificacion.callback_https import EnviadorCallbackHttp
from bv_estimacion.infraestructura.notificacion.segundo_plano import NotificadorEnSegundoPlano
from bv_estimacion.infraestructura.reloj import DormidorAnyio, GeneradorIdsUlid, RelojSistema
from bv_estimacion.infraestructura.trabajos.almacen_temporal import AlmacenTemporalEnDisco
from bv_estimacion.infraestructura.trabajos.ejecutor import EjecutorTrabajos, barrer_periodicamente
from bv_estimacion.infraestructura.trabajos.memoria import ColaTrabajosMemoria, RepositorioTrabajosMemoria

INTERVALO_BARRIDO_S = 60.0  # cada cuánto se purgan los trabajos y las imágenes temporales vencidos

# Conectar, leer y escribir tienen tope propio; el tiempo total de una descarga lo acota el presupuesto del trabajo.
TIEMPOS_HTTP = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)


def create_app(
    configuracion: Configuracion | None = None,
    *,
    fabrica_modelo: ModeloConfigurado | None = None,
) -> FastAPI:
    """`fabrica_modelo` es para los tests: con otra fábrica, la app nunca importa torch."""
    config = configuracion or Configuracion()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # 1. Los pesos son los de la lista, o no se arranca. Leer 130 MB y hashearlos no va en el event loop.
        await to_thread.run_sync(verificar_hashes_modelos, config.carpeta_modelos)
        # 2. Versión y rango salen de config.json (R-20, R-23).
        ficha = cargar_ficha_modelo(config.carpeta_modelos)
        # 3. El modelo, completo, antes de aceptar tráfico.
        modelo = fabrica_modelo or fabrica_real(config.carpeta_modelos)
        adaptador = AdaptadorModelo(
            modelo.crear,
            config.hilos_inferencia,
            ficha.version,
            lado_max_px=modelo.lado_max_px,
            umbral_borrosidad=config.umbral_borrosidad,
        )
        await adaptador.cargar()

        # 4. Todo lo demás.
        reloj = RelojSistema()
        repositorio = RepositorioTrabajosMemoria(config.ttl_trabajos_s, reloj)
        cola = ColaTrabajosMemoria(config.tamano_cola)
        almacen = AlmacenTemporalEnDisco(config.carpeta_temporal, config.ttl_imagenes_s)

        async with httpx.AsyncClient(timeout=TIEMPOS_HTTP) as cliente, anyio.create_task_group() as grupo:
            obtenedor = ObtenedorImagenHttp(cliente, config.hosts_imagen_permitidos, config.max_bytes_imagen)
            enviador = EnviadorCallbackHttp(
                cliente, config.hosts_callback_permitidos, config.secreto_hmac, serializar_trabajo, reloj
            )
            entrega = NotificarResultado(
                enviador,
                DormidorAnyio(),
                intentos=config.intentos_callback,
                espera_base_s=config.espera_base_callback_s,
            )
            notificador = NotificadorEnSegundoPlano(entrega.ejecutar, grupo)
            procesar = ProcesarEstimacion(
                adaptador, repositorio, almacen, obtenedor, notificador, reloj, config.presupuesto_estimacion_s
            )

            # 5. Los workers, recién ahora que el modelo está cargado.
            grupo.start_soon(EjecutorTrabajos(cola, procesar, config.hilos_inferencia).correr)
            grupo.start_soon(
                barrer_periodicamente,
                INTERVALO_BARRIDO_S,
                [repositorio.purgar_vencidos, almacen.barrer_vencidos],
            )

            app.state.contenedor = Contenedor(
                configuracion=config,
                ficha=ficha,
                adaptador=adaptador,
                solicitar=SolicitarEstimacion(repositorio, cola, almacen, GeneradorIdsUlid(), reloj),
                consultar=ConsultarEstimacion(repositorio),
                validar=ValidarImagen(adaptador),
                cliente=cliente,
            )
            try:
                yield
            finally:
                grupo.cancel_scope.cancel()  # apagado: cada worker corta su espera o su trabajo en curso

    app = FastAPI(
        title="BovinoVision · Servicio de estimación de peso",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.configuracion = config
    return app
