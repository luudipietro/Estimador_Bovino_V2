import pytest

from bv_estimacion.dominio.errores import CodigoError, ErrorDeDominio
from bv_estimacion.infraestructura.urls_permitidas import validar_url_permitida

HOSTS = ["almacen.example", "Backend.Example"]


def rechazada(url: str, hosts: list[str] | None = None) -> ErrorDeDominio:
    with pytest.raises(ErrorDeDominio) as error:
        validar_url_permitida(url, HOSTS if hosts is None else hosts)
    assert error.value.codigo is CodigoError.REFERENCIA_NO_PERMITIDA
    return error.value


@pytest.mark.requisito("RF-19")
@pytest.mark.parametrize(
    "url",
    [
        "https://almacen.example/a.jpg?firma=abc",
        "https://ALMACEN.example/a.jpg",  # el host no distingue mayúsculas
        "https://backend.example/cb",  # ni la lista
        "https://almacen.example:8443/a.jpg",
    ],
)
def test_una_url_https_de_un_host_permitido_se_acepta(url: str) -> None:
    validar_url_permitida(url, HOSTS)


@pytest.mark.requisito("RF-19")
@pytest.mark.parametrize(
    "url",
    [
        "http://almacen.example/a.jpg",  # sin TLS
        "ftp://almacen.example/a.jpg",
        "file:///etc/passwd",
        "//almacen.example/a.jpg",  # sin esquema
        "almacen.example/a.jpg",
        "https://otro.example/a.jpg",  # fuera de la lista
        "https://almacen.example.evil.com/a.jpg",  # el host contiene uno permitido, pero no lo es
        "https://evil.com/almacen.example",
        "https://almacen.example@evil.com/a.jpg",  # el host real es evil.com
        "https://usuario:clave@almacen.example/a.jpg",  # credenciales embebidas
        "https://evil.com@almacen.example/a.jpg",  # idem: no se acepta ningún userinfo
        "https://almacen.example:abc/a.jpg",  # puerto inválido
        "https://127.0.0.1/a.jpg",
        "https://localhost/a.jpg",
        "https:///a.jpg",
        "",
    ],
)
def test_las_urls_peligrosas_o_fuera_de_la_allowlist_se_rechazan(url: str) -> None:
    rechazada(url)


@pytest.mark.requisito("RF-19")
def test_con_la_allowlist_vacia_no_se_acepta_ninguna_url() -> None:
    rechazada("https://almacen.example/a.jpg", hosts=[])


@pytest.mark.requisito("RF-19")
def test_el_mensaje_de_error_no_repite_la_url_porque_puede_ir_firmada() -> None:
    error = rechazada("https://otro.example/a.jpg?firma=SECRETO-DE-LA-FIRMA")
    assert "SECRETO-DE-LA-FIRMA" not in str(error)
    assert "otro.example" not in str(error)
