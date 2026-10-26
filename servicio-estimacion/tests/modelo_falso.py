"""Un modelo de mentira y una carpeta de pesos falsos para probar la app sin torch ni los `.pt` reales."""

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from bv_estimacion.infraestructura.config import Configuracion
from bv_estimacion.infraestructura.ml.fabrica import ModeloConfigurado

TOKEN = "token-de-prueba"
SECRETO = "secreto-de-prueba"
LADO_MAX = 1024


def respuesta_modelo(peso: float = 187.4, **cambios: Any) -> dict[str, Any]:
    """Lo que devuelve `EstimadorPeso.estimar` para una foto confiable."""
    base: dict[str, Any] = {
        "peso_kg": peso,
        "confiable": True,
        "motivo": None,
        "confianza_deteccion": 0.94,
        "confianza_keypoint_minima": 0.88,
        "bovinos_en_la_escena": 1,
        "recorte": None,
        "resolucion": "100x100",
        "modelo": "v-prueba",
    }
    return base | cambios


class EstimadorFalso:
    def __init__(self, fabrica: "FabricaFalsa") -> None:
        self._fabrica = fabrica

    def estimar(self, imagen: bytes) -> dict[str, Any]:
        self._fabrica.llamadas.append(imagen)
        if self._fabrica.frenar is not None:
            self._fabrica.frenar.wait(10)
        if self._fabrica.error_al_estimar is not None:
            raise self._fabrica.error_al_estimar
        return self._fabrica.respuestas.get(imagen, self._fabrica.respuesta_por_defecto)


class FabricaFalsa:
    """Reemplaza a `lambda: EstimadorPeso(carpeta)`. Cuenta cuántas instancias se crearon."""

    def __init__(self) -> None:
        self.creadas = 0
        self.llamadas: list[bytes] = []
        self.respuestas: dict[bytes, dict[str, Any]] = {}
        self.respuesta_por_defecto: dict[str, Any] = respuesta_modelo()
        self.error_al_crear: Exception | None = None
        self.error_al_estimar: Exception | None = None
        self.frenar: threading.Event | None = None

    def __call__(self) -> EstimadorFalso:
        if self.error_al_crear is not None:
            raise self.error_al_crear
        self.creadas += 1
        return EstimadorFalso(self)

    def configurada(self) -> ModeloConfigurado:
        return ModeloConfigurado(crear=self, lado_max_px=LADO_MAX)


ARCHIVOS_FALSOS = {"pose.pt": b"pesos de pose", "cnn.pt": b"pesos de la cnn", "det.pt": b"pesos del detector"}


def crear_carpeta_modelo(carpeta: Path, **config: object) -> Path:
    """Una carpeta con `config.json`, `SHA256SUMS.txt` y archivos falsos cuyos hashes coinciden."""
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre, contenido in ARCHIVOS_FALSOS.items():
        (carpeta / nombre).write_bytes(contenido)
    suma = "".join(f"{hashlib.sha256(c).hexdigest()} *{n}\n" for n, c in ARCHIVOS_FALSOS.items())
    (carpeta / "SHA256SUMS.txt").write_text(suma, encoding="utf-8")
    base: dict[str, object] = {
        "version": "v-prueba / datos de prueba",
        "rango_entrenamiento_kg": [36, 621],
        "archivo_pose": "pose.pt",
        "archivo_cnn": "cnn.pt",
        "archivo_detector": "det.pt",
    }
    (carpeta / "config.json").write_text(json.dumps(base | config), encoding="utf-8")
    return carpeta


def configuracion_de_prueba(tmp_path: Path, **cambios: Any) -> Configuracion:
    """La configuración de una app de prueba: carpetas dentro de `tmp_path`, un solo hilo y sin leer `.env`."""
    valores: dict[str, Any] = {
        "token_servicio": TOKEN,
        "secreto_hmac": SECRETO,
        "carpeta_modelos": crear_carpeta_modelo(tmp_path / "modelos"),
        "carpeta_temporal": tmp_path / "temporal",
        "hilos_inferencia": 1,
        "presupuesto_estimacion_s": 5.0,
        "hosts_imagen_permitidos": ["almacenamiento.example"],
        "hosts_callback_permitidos": ["backend.example"],
        "intentos_callback": 1,
        "espera_base_callback_s": 0.0,
    }
    return Configuracion(_env_file=None, **(valores | cambios))  # type: ignore[call-arg]  # _env_file es de pydantic-settings
