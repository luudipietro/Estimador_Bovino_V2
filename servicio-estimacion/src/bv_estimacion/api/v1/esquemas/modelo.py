from typing import Literal

from pydantic import BaseModel, Field


class Salud(BaseModel):
    estado: Literal["ok", "no_disponible"]
    modelo_cargado: bool
    version_modelo: str | None
    instancias: int = Field(description="Instancias del modelo cargadas (una por hilo de inferencia)")
    error: str | None = Field(description="Por qué no está disponible, o `null`")


class FichaModeloRespuesta(BaseModel):
    version: str
    rango_validez_kg: tuple[float, float] = Field(description="Fuera de este rango el peso no es confiable (R-20)")
    mape_test: float | None = Field(description="`null` si el `config.json` del modelo no lo informa")
    ic95_test: tuple[float, float] | None
    fecha_entrenamiento: str | None = Field(description="`null` si el `config.json` del modelo no la informa")
