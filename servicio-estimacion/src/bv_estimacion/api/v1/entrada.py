"""Lectura y validación **en el borde** de lo que llega: encabezado, cuerpo, tamaños, tipos reales, allowlists. Todo se
rechaza acá, antes de guardar nada ni encolar. No hay lógica de negocio: eso es de los casos de uso."""

import json
import re
from functools import partial
from typing import Any

from anyio import to_thread
from fastapi import Request
from pydantic import TypeAdapter, ValidationError
from starlette.datastructures import FormData, UploadFile

from bv_estimacion.api.errores import SolicitudInvalida
from bv_estimacion.api.v1.esquemas.pedido import PedidoPorReferencias, ReferenciaImagenIn
from bv_estimacion.aplicacion.dto import ReferenciaExterna, SolicitarEstimacionCmd
from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio
from bv_estimacion.dominio.estimacion import MAX_IMAGENES, MIN_IMAGENES
from bv_estimacion.dominio.valores import IdEvento
from bv_estimacion.infraestructura.config import Configuracion
from bv_estimacion.infraestructura.imagenes.lectura import leer_con_limite
from bv_estimacion.infraestructura.imagenes.validacion import validar_imagen
from bv_estimacion.infraestructura.urls_permitidas import validar_url_permitida

ID_EVENTO = re.compile(r"[\x21-\x7e]{1,128}")  # ASCII imprimible, sin espacios
MAX_PARTES = 20  # más que MAX_IMAGENES: de 5 a 20 es "cantidad inválida" (422); más que eso, un multipart absurdo (400)
CAMPOS_ESTIMACION = {"imagenes", "referencias_imagen", "metadatos", "url_notificacion"}
_REFERENCIAS = TypeAdapter(list[ReferenciaImagenIn])
_MENSAJE_CANTIDAD = (
    f"Se esperaban entre {MIN_IMAGENES} y {MAX_IMAGENES} imágenes, por multipart o por referencia, "
    "pero no las dos formas a la vez."
)


def _tipo_de_contenido(request: Request) -> str:
    return request.headers.get("content-type", "").split(";")[0].strip().lower()


def _campos_de(error: ValidationError) -> list[str]:
    return sorted({".".join(str(parte) for parte in e["loc"]) or "cuerpo" for e in error.errors()})


def _cantidad_invalida() -> ErrorDeDominio:
    return ErrorDeDominio(CodigoError.CANTIDAD_IMAGENES_INVALIDA, _MENSAJE_CANTIDAD, regla="R-18")


def leer_id_evento(idempotency_key: str | None) -> IdEvento:
    """`Idempotency-Key` = `id_evento`. Es obligatorio y viene solo en el encabezado."""
    if idempotency_key is None or not ID_EVENTO.fullmatch(idempotency_key):
        raise SolicitudInvalida(
            "Falta el encabezado Idempotency-Key o no es válido (1 a 128 caracteres ASCII imprimibles, sin espacios).",
            ["Idempotency-Key"],
        )
    return IdEvento(idempotency_key)


async def _formulario(request: Request) -> FormData:
    if _tipo_de_contenido(request) != "multipart/form-data":
        raise SolicitudInvalida("El Content-Type tiene que ser multipart/form-data.", ["Content-Type"])
    return await request.form(max_files=MAX_PARTES, max_fields=MAX_PARTES)


async def _leer_archivo(archivo: UploadFile, config: Configuracion) -> bytes:
    """Lee con tope de tamaño (corta apenas se pasa) y valida formato real, decodificabilidad y resolución."""
    datos = await leer_con_limite(archivo.read, config.max_bytes_imagen)
    await to_thread.run_sync(  # decodificar una foto es CPU: fuera del event loop
        partial(
            validar_imagen,
            datos,
            max_bytes=config.max_bytes_imagen,
            max_pixeles=config.max_pixeles,
            resolucion_minima_px=config.resolucion_minima_px,
        )
    )
    return datos


def _metadatos_de_texto(texto: str | None) -> dict[str, Any]:
    if texto is None:
        return {}
    try:
        valor = json.loads(texto)
    except ValueError as error:
        raise SolicitudInvalida("metadatos tiene que ser un objeto JSON.", ["metadatos"]) from error
    if not isinstance(valor, dict):
        raise SolicitudInvalida("metadatos tiene que ser un objeto JSON.", ["metadatos"])
    return valor


def _referencias_de_texto(texto: str) -> list[ReferenciaImagenIn]:
    try:
        return _REFERENCIAS.validate_json(texto)
    except ValidationError as error:
        raise SolicitudInvalida("referencias_imagen está mal formado.", ["referencias_imagen"]) from error


def _validar_urls(config: Configuracion, referencias: list[ReferenciaImagenIn], url_notificacion: str | None) -> None:
    for referencia in referencias:
        validar_url_permitida(referencia.url, config.hosts_imagen_permitidos)
    if url_notificacion is not None:
        validar_url_permitida(url_notificacion, config.hosts_callback_permitidos)


def _externas(referencias: list[ReferenciaImagenIn]) -> tuple[ReferenciaExterna, ...]:
    return tuple(ReferenciaExterna(r.url, r.sha256.lower()) for r in referencias)


async def leer_pedido_de_estimacion(
    request: Request, config: Configuracion, id_evento: IdEvento
) -> SolicitarEstimacionCmd:
    tipo = _tipo_de_contenido(request)
    if tipo == "multipart/form-data":
        return await _desde_multipart(request, config, id_evento)
    if tipo == "application/json":
        return await _desde_json(request, config, id_evento)
    raise SolicitudInvalida("El Content-Type tiene que ser multipart/form-data o application/json.", ["Content-Type"])


async def _desde_multipart(request: Request, config: Configuracion, id_evento: IdEvento) -> SolicitarEstimacionCmd:
    formulario = await _formulario(request)
    try:
        archivos: list[UploadFile] = []
        texto: dict[str, str] = {}
        for nombre, valor in formulario.multi_items():
            if nombre not in CAMPOS_ESTIMACION:
                raise SolicitudInvalida("Hay un campo desconocido.", [nombre])
            if nombre == "imagenes":
                if not isinstance(valor, UploadFile):
                    raise SolicitudInvalida("imagenes tiene que ser un archivo.", ["imagenes"])
                archivos.append(valor)
            elif isinstance(valor, UploadFile) or nombre in texto:
                raise SolicitudInvalida(f"{nombre} tiene que ser un único campo de texto.", [nombre])
            else:
                texto[nombre] = valor

        metadatos = _metadatos_de_texto(texto.get("metadatos"))
        referencias = _referencias_de_texto(texto["referencias_imagen"]) if "referencias_imagen" in texto else []
        url_notificacion = texto.get("url_notificacion")

        # Lo barato primero: nada de decodificar fotos si el pedido ya está mal.
        if (archivos and referencias) or not MIN_IMAGENES <= len(archivos) + len(referencias) <= MAX_IMAGENES:
            raise _cantidad_invalida()
        _validar_urls(config, referencias, url_notificacion)

        return SolicitarEstimacionCmd(
            id_evento=id_evento,
            imagenes=tuple([await _leer_archivo(archivo, config) for archivo in archivos]),
            referencias=_externas(referencias),
            metadatos=metadatos,
            url_notificacion=url_notificacion,
        )
    finally:
        await formulario.close()


async def _desde_json(request: Request, config: Configuracion, id_evento: IdEvento) -> SolicitarEstimacionCmd:
    try:
        cuerpo = json.loads(await request.body())
    except ValueError as error:
        raise SolicitudInvalida("El cuerpo no es un JSON válido.") from error
    if not isinstance(cuerpo, dict):
        raise SolicitudInvalida("El cuerpo tiene que ser un objeto JSON.")
    if "imagenes" in cuerpo:
        raise _cantidad_invalida()  # las imágenes van por multipart; el JSON solo lleva referencias

    try:
        pedido = PedidoPorReferencias.model_validate(cuerpo)
    except ValidationError as error:
        raise SolicitudInvalida("El pedido está mal formado.", _campos_de(error)) from error

    if not MIN_IMAGENES <= len(pedido.referencias_imagen) <= MAX_IMAGENES:
        raise _cantidad_invalida()
    _validar_urls(config, pedido.referencias_imagen, pedido.url_notificacion)

    return SolicitarEstimacionCmd(
        id_evento=id_evento,
        referencias=_externas(pedido.referencias_imagen),
        metadatos=pedido.metadatos or {},
        url_notificacion=pedido.url_notificacion,
    )


async def leer_imagen_de_validacion(request: Request, config: Configuracion) -> bytes:
    """`POST /v1/validaciones`: multipart con exactamente un archivo en el campo `imagen`."""
    formulario = await _formulario(request)
    try:
        items = list(formulario.multi_items())
        desconocidos = [nombre for nombre, _ in items if nombre != "imagen"]
        if desconocidos:
            raise SolicitudInvalida("Hay un campo desconocido.", desconocidos)
        archivos = [valor for _, valor in items if isinstance(valor, UploadFile)]
        if len(archivos) != 1 or len(items) != 1:
            raise SolicitudInvalida("Hace falta exactamente un archivo en el campo imagen.", ["imagen"])
        return await _leer_archivo(archivos[0], config)
    finally:
        await formulario.close()
