"""Autenticación de servicio a servicio: `Authorization: Bearer <token>`. Solo `/salud` es abierta.

Se aplica por ruta (`dependencies=[Depends(exigir_token)]` en cada router) y no como filtro global, así que un
router nuevo tiene que pedirla: hay un test que recorre todas las rutas y falla si alguna queda abierta.
"""

import secrets

from fastapi import Request

from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio

# Mismo mensaje para "no vino" y "no es": no se le dice a quien prueba cuál de las dos fue.
MENSAJE = "Falta el token de servicio o no es válido."


async def exigir_token(request: Request) -> None:
    esperado: str = request.app.state.configuracion.token_servicio
    esquema, _, token = request.headers.get("authorization", "").partition(" ")
    if esquema.lower() == "bearer" and token and secrets.compare_digest(token.encode(), esperado.encode()):
        return
    raise ErrorDeDominio(CodigoError.NO_AUTENTICADO, MENSAJE)
