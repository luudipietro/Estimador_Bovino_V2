"""Configuración del servicio, leída de variables de entorno. La tabla completa está en `docs/arquitectura.md`.

Nada de acá se versiona con valores reales: el repo solo trae `.env.example`, sin secretos.
"""

import os
import tempfile
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _hilos_por_defecto() -> int:
    return os.cpu_count() or 1


def _carpeta_temporal_por_defecto() -> Path:
    return Path(tempfile.gettempdir()) / "bv_estimacion"


class Configuracion(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Obligatorias (sin valor por defecto: el servicio no arranca sin ellas) ---
    token_servicio: str = Field(min_length=1)
    secreto_hmac: str = Field(min_length=1)

    # --- Allowlists (anti-SSRF). Vacías = no se acepta ninguna URL. Hosts separados por coma. ---
    hosts_imagen_permitidos: Annotated[list[str], NoDecode] = Field(default_factory=list)
    hosts_callback_permitidos: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # --- Modelo y trabajo ---
    carpeta_modelos: Path = Path("../estimador/modelos")
    presupuesto_estimacion_s: float = Field(default=45.0, gt=0)  # RNF-01 (propuesto)
    hilos_inferencia: int = Field(default_factory=_hilos_por_defecto, ge=1)

    # --- Imágenes ---
    max_mb_imagen: int = Field(default=25, ge=1)
    max_pixeles: int = Field(default=100_000_000, ge=1)  # propuesto: protege de bombas de descompresión
    resolucion_minima_mp: float = Field(default=3.0, gt=0)  # a confirmar con el equipo
    carpeta_temporal: Path = Field(default_factory=_carpeta_temporal_por_defecto)
    ttl_imagenes_s: int = Field(default=3600, ge=1)  # propuesto: respaldo si el proceso cae a mitad de camino

    # --- Calidad (R-24). None = la regla no se aplica hasta que se calibre ---
    umbral_borrosidad: float | None = None

    # --- Trabajos y cola ---
    ttl_trabajos_s: int = Field(default=86_400, ge=1)  # propuesto: 24 h
    tamano_cola: int = Field(default=200, ge=1)  # propuesto

    # --- Callback ---
    intentos_callback: int = Field(default=3, ge=1)
    espera_base_callback_s: float = Field(default=1.0, ge=0)

    @field_validator("hosts_imagen_permitidos", "hosts_callback_permitidos", mode="before")
    @classmethod
    def _separar_hosts(cls, valor: object) -> object:
        if isinstance(valor, str):
            return [h.strip().lower() for h in valor.split(",") if h.strip()]
        return valor

    @field_validator("umbral_borrosidad", mode="before")
    @classmethod
    def _vacio_es_none(cls, valor: object) -> object:
        # `UMBRAL_BORROSIDAD=` en el .env significa "sin umbral", no un número inválido
        if isinstance(valor, str) and not valor.strip():
            return None
        return valor

    @property
    def max_bytes_imagen(self) -> int:
        return self.max_mb_imagen * 1024 * 1024

    @property
    def resolucion_minima_px(self) -> int:
        return round(self.resolucion_minima_mp * 1_000_000)
