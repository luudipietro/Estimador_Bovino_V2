"""Probar el modelo con fotos de vacas, sin la API ni el backend. Un solo archivo, usa `estimador.py` tal cual.

Tres formas de usarlo (desde la carpeta `estimador/`, con el entorno que ya tiene torch):

  1) Página web local: arrastrás fotos y ves el peso, el recorte que usó el modelo y, si escribís el peso real, el error.
         .venv\\Scripts\\python probar.py --web

  2) Unas fotos sueltas, por consola:
         .venv\\Scripts\\python probar.py vaca1.jpg vaca2.jpg

  3) Evaluar una carpeta con los pesos reales (MAPE contra predecir siempre la mediana):
         .venv\\Scripts\\python probar.py --carpeta fotos_test --pesos pesos.csv
     `pesos.csv` tiene una fila por foto: `foto,peso_kg` (y opcionalmente `animal`, para promediar las fotos de un animal).

Por defecto compara los dos modelos que haya: `v1` (carpeta `modelos`, entrenado en Bangladesh) y `v2` (carpeta
`modelos_v2`, ajustado con el dataset propio). Se puede elegir otra cosa con `--modelo nombre=carpeta` (repetible).

Detalles que importan:
  - Aplica la orientación EXIF antes de estimar (el iPhone guarda las fotos verticales "acostadas").
  - El rango de validez sale del `config.json` de cada modelo (no del que trae escrito `estimador.py`): un peso fuera de ese
    rango se informa pero NO se marca como confiable (R-20). El v1 declara 36-621 kg pero no pasa de ~270, así que sobre
    vacas pesadas da valores bajos igual "dentro del rango".
  - La página solo escucha en esta máquina (127.0.0.1): nadie más en la red puede usarla.
"""

import argparse
import base64
import csv
import io
import json
import statistics
import sys
import threading
import time
import webbrowser
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

AQUI = Path(__file__).resolve().parent
TEXTO_FUERA_DE_RANGO_DEL_MODULO = "el peso estimado cae fuera del rango con el que se entrenó el modelo"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".heic"}
MAX_LADO_MINIATURA = 900

try:  # HEIC (fotos originales del iPhone 14): opcional
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass


class Modelo:
    """Un modelo cargado: el `EstimadorPeso` y lo que declara su `config.json`."""

    def __init__(self, nombre: str, carpeta: Path) -> None:
        from estimador import EstimadorPeso  # importa PyTorch: tarda unos segundos

        self.nombre = nombre
        self.carpeta = carpeta
        config = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
        self.version = str(config.get("version", "sin versión"))
        self.rango = (float(config["rango_entrenamiento_kg"][0]), float(config["rango_entrenamiento_kg"][1]))
        self.estimador = EstimadorPeso(carpeta)
        self._candado = threading.Lock()  # los modelos de Ultralytics no son seguros con varios hilos a la vez

    def estimar(self, imagen: Image.Image) -> dict:
        inicio = time.perf_counter()
        with self._candado:
            r = self.estimador.estimar(imagen)
        peso = r.get("peso_kg")
        motivos = [m for m in (r.get("motivo") or "").split("; ") if m and m != TEXTO_FUERA_DE_RANGO_DEL_MODULO]
        if peso is not None and not (self.rango[0] <= peso <= self.rango[1]):
            motivos.append(f"fuera del rango del modelo ({self.rango[0]:.0f}-{self.rango[1]:.0f} kg)")
        if peso is None and not motivos:
            motivos.append("no se detectó ningún bovino en la foto")
        return {
            "peso_kg": peso,
            "confiable": peso is not None and not motivos,
            "motivos": motivos,
            "confianza_deteccion": r.get("confianza_deteccion"),
            "keypoints": r.get("keypoints"),
            "recorte": r.get("recorte"),
            "ms": round((time.perf_counter() - inicio) * 1000),
        }


def cargar_modelos(pedidos: list[str]) -> list[Modelo]:
    if not pedidos:
        pedidos = [f"{n}={c}" for n, c in (("v1", "modelos"), ("v2", "modelos_v2")) if (AQUI / c / "config.json").exists()]
    modelos = []
    for pedido in pedidos:
        nombre, _, carpeta = pedido.partition("=")
        ruta = (Path(carpeta) if Path(carpeta).is_absolute() else AQUI / carpeta)
        if not (ruta / "config.json").exists():
            sys.exit(f"no encuentro {ruta / 'config.json'} (¿bajaste los modelos? ver README.md)")
        print(f"cargando {nombre} ({ruta.name})...", flush=True)
        modelos.append(Modelo(nombre, ruta))
    if not modelos:
        sys.exit("no hay ningún modelo: pasá --modelo nombre=carpeta")
    return modelos


def abrir_foto(origen: Path | bytes) -> Image.Image:
    abierta = Image.open(origen if isinstance(origen, Path) else io.BytesIO(origen))
    return ImageOps.exif_transpose(abierta).convert("RGB")


def miniatura_anotada(imagen: Image.Image, resultado: dict) -> str:
    """La foto con el recorte que usó el modelo (verde) y los 9 puntos (rojo), en base64 para la página."""
    mayor = max(imagen.size)
    vista = imagen.copy() if mayor <= 1024 else imagen.resize((round(imagen.width * 1024 / mayor), round(imagen.height * 1024 / mayor)))
    dibujo = ImageDraw.Draw(vista)
    if resultado.get("recorte"):
        dibujo.rectangle(resultado["recorte"], outline=(40, 200, 80), width=4)
    for x, y, confianza in (resultado.get("keypoints") or {}).values():
        color = (230, 40, 40) if confianza >= 0.5 else (240, 180, 0)
        dibujo.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline=(255, 255, 255))
    vista.thumbnail((MAX_LADO_MINIATURA, MAX_LADO_MINIATURA))
    buffer = io.BytesIO()
    vista.save(buffer, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


# --------------------------------------------------------------------------------------------------- consola


def leer_pesos(ruta: Path) -> dict[str, tuple[float, str | None]]:
    """`foto -> (peso_kg, animal)`. Acepta `foto|archivo`, `peso_kg|peso` y `animal` (opcional)."""
    pesos: dict[str, tuple[float, str | None]] = {}
    with ruta.open(encoding="utf-8-sig", newline="") as f:
        for fila in csv.DictReader(f):
            foto = fila.get("foto") or fila.get("archivo")
            peso = fila.get("peso_kg") or fila.get("peso")
            if foto and peso:
                pesos[foto.strip()] = (float(str(peso).replace(",", ".")), (fila.get("animal") or "").strip() or None)
    return pesos


def mape(real: list[float], pred: list[float]) -> float:
    return sum(abs(p - r) / r for r, p in zip(real, pred, strict=True)) / len(real) * 100


def correr_consola(
    modelos: list[Modelo], fotos: list[Path], pesos: dict[str, tuple[float, str | None]], referencia: float | None = None
) -> None:
    for m in modelos:
        print(f"\n=== {m.nombre}: {m.version}  (rango válido {m.rango[0]:.0f}-{m.rango[1]:.0f} kg)")
        filas = []
        for foto in fotos:
            try:
                r = m.estimar(abrir_foto(foto))
            except Exception as error:  # una foto ilegible no frena a las demás
                print(f"  {foto.name:<42} no se pudo abrir ({type(error).__name__})")
                continue
            real = pesos.get(foto.name, (None, None))
            filas.append((foto.name, real[0], real[1], r))
            estado = "ok" if r["confiable"] else "NO SÉ: " + "; ".join(r["motivos"])
            estimado = "      -" if r["peso_kg"] is None else f"{r['peso_kg']:7.1f}"
            extra = ""
            if real[0] is not None and r["peso_kg"] is not None:
                extra = f"  real {real[0]:6.1f}  error {abs(r['peso_kg'] - real[0]) / real[0] * 100:5.1f}%"
            print(f"  {foto.name:<42} {estimado} kg  {estado}{extra}")
        resumir(filas, referencia)


def resumir(filas: list[tuple[str, float | None, str | None, dict]], referencia: float | None = None) -> None:
    con_real = [(n, real, animal, r) for n, real, animal, r in filas if real is not None and r["peso_kg"] is not None]
    if not con_real:
        return
    reales = [real for _, real, _, _ in con_real]
    preds = [r["peso_kg"] for _, _, _, r in con_real]
    origen = "fijada por vos" if referencia is not None else "la mediana de ESTA muestra"
    referencia = referencia if referencia is not None else statistics.median(reales)
    print(f"\n  Resumen sobre {len(con_real)} fotos con peso real:")
    print(f"    MAPE del modelo                    : {mape(reales, preds):5.1f} %")
    print(f"    MAPE de predecir siempre {referencia:.0f} kg    : {mape(reales, [referencia] * len(reales)):5.1f} %   ({origen})")
    print(f"    sesgo medio (estimado - real)      : {statistics.mean(p - r for r, p in zip(reales, preds, strict=True)):+5.1f} kg")
    errores = [abs(p - r) / r * 100 for r, p in zip(reales, preds, strict=True)]
    print(f"    fotos con error <= 10 %            : {sum(e <= 10 for e in errores)} de {len(errores)}")
    confiables = [(r_, p) for (_, r_, _, r), p in zip(con_real, preds, strict=True) if r["confiable"]]
    print(f"    el modelo dijo 'confiable' en      : {len(confiables)} de {len(con_real)} fotos")
    por_animal: dict[str, list[tuple[float, float]]] = {}
    for _, real, animal, r in con_real:
        if animal:
            por_animal.setdefault(animal, []).append((real, r["peso_kg"]))
    if por_animal:
        import math

        reales_a = [v[0][0] for v in por_animal.values()]
        preds_a = [math.exp(statistics.mean(math.log(p) for _, p in v)) for v in por_animal.values()]
        print(f"    MAPE por animal ({len(por_animal)} animales, fotos promediadas): {mape(reales_a, preds_a):5.1f} %")


# --------------------------------------------------------------------------------------------------- página web

PAGINA = """<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Probar el modelo de peso</title>
<style>
 body{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:#f4f1ea;color:#222}
 header{background:#2f4a2b;color:#fff;padding:14px 22px} header h1{margin:0;font-size:20px}
 main{max-width:1200px;margin:18px auto;padding:0 16px}
 .info{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px}
 .modelo{background:#fff;border-radius:8px;padding:8px 12px;font-size:13px;border-left:5px solid #7a9b57}
 #zona{border:3px dashed #7a9b57;border-radius:12px;padding:30px;text-align:center;background:#fff;cursor:pointer}
 #zona.sobre{background:#e6f0da} #zona p{margin:4px 0;color:#555}
 table{width:100%;border-collapse:collapse;margin-top:16px;background:#fff;border-radius:8px;overflow:hidden}
 th,td{padding:8px 10px;border-bottom:1px solid #eee;text-align:left;vertical-align:middle;font-size:14px}
 th{background:#e9e4d6} img{max-width:230px;border-radius:6px;display:block}
 .peso{font-size:22px;font-weight:700} .ok{color:#1b7a2b} .nose{color:#b3261e;font-size:12px}
 .aviso{color:#8a6d00;font-size:12px} input[type=number]{width:80px;padding:5px}
 #resumen{margin-top:14px;background:#fff;padding:10px 14px;border-radius:8px;font-size:14px;display:none}
 .err{font-weight:600}
</style>
<header><h1>Probar el modelo de peso con fotos de vacas</h1></header>
<main>
 <div class="info" id="info"></div>
 <div id="zona"><p><b>Arrastrá fotos de vacas acá</b> o hacé clic para elegirlas</p>
  <p>Foto lateral, con el animal completo. Si conocés el peso real, escribilo al lado para ver el error.</p>
  <input type="file" id="archivos" multiple accept="image/*,.heic" hidden></div>
 <div id="resumen"></div>
 <table id="tabla" hidden><thead><tr id="cabecera"><th>Foto (verde = recorte, rojo = puntos)</th></tr></thead><tbody></tbody></table>
</main>
<script>
let modelos = [], filas = [];
const $ = (id) => document.getElementById(id);
async function iniciar() {
  modelos = await (await fetch('/info')).json();
  $('info').innerHTML = modelos.map(m => `<div class="modelo"><b>${m.id}</b>: ${m.version}<br>rango válido ${m.rango[0]}-${m.rango[1]} kg</div>`).join('');
  $('cabecera').innerHTML += modelos.map(m => `<th>${m.id}: peso estimado</th>`).join('') + '<th>Peso real (kg)</th>' + modelos.map(m => `<th>${m.id}: error</th>`).join('');
}
function celda(r) {
  if (!r) return '<td>...</td>';
  const peso = r.peso_kg === null ? '-' : r.peso_kg.toFixed(1) + ' kg';
  const estado = r.confiable ? '<span class="ok">confiable</span>' : '<span class="nose">NO SÉ: ' + r.motivos.join('; ') + '</span>';
  return `<td><div class="peso">${peso}</div>${estado}<div class="aviso">${r.ms} ms</div></td>`;
}
function agregar(archivo) {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td><div>${archivo.name.replace(/</g, '&lt;')}</div><div class="aviso">procesando...</div></td>` + modelos.map(() => '<td>...</td>').join('') + '<td></td>' + modelos.map(() => '<td></td>').join('');
  $('tabla').hidden = false; document.querySelector('#tabla tbody').appendChild(tr);
  const fila = {tr, resultados: null, real: null}; filas.push(fila); return fila;
}
function pintar(fila, datos) {
  const celdas = fila.tr.children;
  if (datos.error) { celdas[0].innerHTML += `<div class="nose">${datos.error}</div>`; return; }
  fila.resultados = datos.modelos;
  celdas[0].innerHTML = `<img src="${datos.imagen}" title="${datos.nombre}"><div class="aviso">${datos.nombre}</div>` + datos.advertencias.map(a => `<div class="aviso">${a}</div>`).join('');
  modelos.forEach((m, i) => { celdas[1 + i].outerHTML = celda(datos.modelos[m.id]); });
  const c = fila.tr.children; const idx = 1 + modelos.length;
  c[idx].innerHTML = '<input type="number" min="1" step="0.1" placeholder="kg">';
  c[idx].firstChild.addEventListener('input', (e) => { fila.real = parseFloat(e.target.value) || null; recalcular(); });
}
function recalcular() {
  const idx = 1 + modelos.length;
  const resumen = modelos.map(() => ({n: 0, suma: 0}));
  filas.forEach(f => {
    if (!f.resultados) return;
    modelos.forEach((m, i) => {
      const r = f.resultados[m.id], celda = f.tr.children[idx + 1 + i];
      if (f.real && r && r.peso_kg !== null) {
        const e = Math.abs(r.peso_kg - f.real) / f.real * 100;
        celda.innerHTML = `<span class="err">${e.toFixed(1)} %</span><div class="aviso">${(r.peso_kg - f.real >= 0 ? '+' : '') + (r.peso_kg - f.real).toFixed(1)} kg</div>`;
        resumen[i].n++; resumen[i].suma += e;
      } else celda.innerHTML = '';
    });
  });
  const hay = resumen.some(r => r.n);
  $('resumen').style.display = hay ? 'block' : 'none';
  $('resumen').innerHTML = '<b>Error medio (MAPE) con los pesos que cargaste:</b> ' + modelos.map((m, i) => resumen[i].n ? `${m.id}: <b>${(resumen[i].suma / resumen[i].n).toFixed(1)} %</b> (${resumen[i].n} fotos)` : `${m.id}: -`).join(' &nbsp;|&nbsp; ') + '<div class="aviso">Referencia: predecir siempre el peso medio del grupo suele dar 8-9 % en un rebaño parejo; un MAPE parecido no es mejora.</div>';
}
async function procesar(lista) {
  for (const archivo of lista) {
    const fila = agregar(archivo);
    const fd = new FormData(); fd.append('foto', archivo);
    try { pintar(fila, await (await fetch('/estimar', {method: 'POST', body: fd})).json()); }
    catch (e) { pintar(fila, {error: 'falló la conexión con el servidor local'}); }
  }
}
$('zona').addEventListener('click', () => $('archivos').click());
$('archivos').addEventListener('change', (e) => procesar([...e.target.files]));
['dragover', 'dragenter'].forEach(ev => $('zona').addEventListener(ev, (e) => { e.preventDefault(); $('zona').classList.add('sobre'); }));
['dragleave', 'drop'].forEach(ev => $('zona').addEventListener(ev, (e) => { e.preventDefault(); $('zona').classList.remove('sobre'); }));
$('zona').addEventListener('drop', (e) => procesar([...e.dataTransfer.files]));
iniciar();
</script></html>"""


def correr_web(modelos: list[Modelo], puerto: int, abrir: bool) -> None:
    import uvicorn
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="Probar el modelo de peso")

    @app.get("/", response_class=HTMLResponse)
    def inicio() -> str:
        return PAGINA

    @app.get("/info")
    def info() -> list[dict]:
        return [{"id": m.nombre, "version": m.version, "rango": list(m.rango)} for m in modelos]

    @app.post("/estimar")
    def estimar(foto: UploadFile = File(...)) -> dict:  # `def`: FastAPI lo corre en un hilo, no bloquea el servidor
        try:
            imagen = abrir_foto(foto.file.read())
        except Exception:
            return {"error": "no se pudo abrir la imagen (si es HEIC, hace falta pillow-heif)"}
        resultados = {m.nombre: m.estimar(imagen) for m in modelos}
        primero = next(iter(resultados.values()))
        return {
            "nombre": foto.filename,
            "imagen": miniatura_anotada(imagen, primero),
            "advertencias": [] if imagen.getexif() else ["la foto no trae metadata EXIF"],
            "modelos": {k: {c: v for c, v in r.items() if c not in ("keypoints", "recorte")} for k, r in resultados.items()},
        }

    url = f"http://127.0.0.1:{puerto}"
    print(f"\nListo. Abrí {url}   (Ctrl+C para cortar)")
    if abrir:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=puerto, log_level="warning")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fotos", nargs="*", type=Path, help="fotos a estimar (consola)")
    ap.add_argument("--web", action="store_true", help="abrir la página local para arrastrar fotos")
    ap.add_argument("--puerto", type=int, default=8002)
    ap.add_argument("--no-abrir", action="store_true", help="con --web: no abrir el navegador solo")
    ap.add_argument("--carpeta", type=Path, help="estimar todas las fotos de esta carpeta")
    ap.add_argument("--pesos", type=Path, help="CSV con `foto,peso_kg` para calcular el error")
    ap.add_argument("--referencia", type=float, help="peso fijo (kg) para la referencia 'predecir siempre lo mismo'")
    ap.add_argument("--modelo", action="append", default=[], metavar="NOMBRE=CARPETA", help="repetible")
    args = ap.parse_args()

    modelos = cargar_modelos(args.modelo)
    if args.web:
        correr_web(modelos, args.puerto, not args.no_abrir)
        return

    fotos = list(args.fotos)
    if args.carpeta:
        fotos += sorted(p for p in args.carpeta.iterdir() if p.suffix.lower() in EXTENSIONES)
    if not fotos:
        ap.error("no hay fotos: pasá archivos, --carpeta o --web")
    correr_consola(modelos, fotos, leer_pesos(args.pesos) if args.pesos else {}, args.referencia)


if __name__ == "__main__":
    main()
