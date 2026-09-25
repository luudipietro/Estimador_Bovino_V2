# Qué necesitamos del campo para armar el dataset propio

Documento para coordinar las visitas. Todo lo de acá sale de lo que ya medimos con datasets externos
(4.544 fotos de Bangladesh, 513 de CID, 72 de Mongolia); las decisiones están justificadas con esos números.

---

## 1. Lo esencial, en una frase

Por cada animal necesitamos: **foto lateral + peso real de balanza + identificación del animal**, tomados
**el mismo día**. Nada de marcadores ni accesorios sobre el animal.

---

## 2. Cuántos animales

La relación que medimos entre cantidad de datos y error, por banda de peso:

| Animales distintos en la banda | Error esperable en esa banda |
|---|---|
| ~20 | 27% |
| ~50 | 22% |
| ~215 | 15,6% |
| ~340 | 11,8% |

**El error no baja por juntar más fotos del mismo animal, sino por sumar animales distintos.**

### Meta recomendada

| Prioridad | Animales | Para qué alcanza |
|---|---|---|
| **Mínimo viable** | **150–200** en total, repartidos | Demostrar que funciona; error esperable 15–20% |
| **Objetivo** | **300–400** | Error esperable 12–15%; es la meta realista del proyecto |
| Ideal | 500+ | Margen para mejorar los extremos |

### Cómo repartirlos por peso

Hay que cubrir **todo el rango que el sistema va a tener que estimar**, porque el modelo **no extrapola**:
lo comprobamos, nunca predice fuera del rango con el que fue entrenado. Si solo le damos animales de 400–500 kg,
va a fallar con un ternero, y al revés.

Reparto sugerido para 300 animales (ajustar a lo que realmente tenga el establecimiento):

| Banda | Animales | Comentario |
|---|---|---|
| menos de 150 kg | 50 | Terneros. Es la banda más difícil: el error relativo siempre es mayor |
| 150–250 kg | 60 | |
| 250–350 kg | 70 | |
| **350–450 kg** | **70** | Zona comercial, prioridad |
| **450–550 kg** | **50** | Zona comercial, prioridad |
| más de 550 kg | los que haya | Vacas y toros adultos |

**Ninguna banda debería quedar con menos de 40–50 animales**, o el error en esa banda se dispara.

### Fotos por animal
**2 a 4 fotos laterales** del mismo animal en la misma sesión, variando un poco el ángulo y la posición del
operario. No reemplazan a tener más animales, pero hacen el modelo más robusto y permiten promediar predicciones.

### Un mismo animal en varias fechas
**Muy valioso.** Si el establecimiento pesa los mismos animales cada 30–60 días, registrar esas repeticiones
multiplica el valor del dataset: el modelo aprende a ver el cambio de peso del mismo animal, que es justo lo
que el productor quiere medir. En el dataset de Bangladesh, el mismo animal aparecía con hasta 106 kg de
diferencia entre fotos, y eso resultó ser una de sus mayores fortalezas.

---

## 3. Protocolo de captura

### Lo que sí hay que hacer
1. **Vista lateral**, el animal completo dentro del cuadro (de la cruz a las pezuñas, y de pecho a cola).
2. **Cámara perpendicular al animal**, a la altura del lomo aproximadamente, no desde arriba ni desde abajo.
3. **El animal parado y con las cuatro patas apoyadas**, lo más derecho posible.
4. **Encuadre consistente**: que el animal ocupe más o menos lo mismo en todas las fotos (alrededor de dos
   tercios del ancho). No hace falta medir la distancia, pero conviene no variar entre "muy de cerca" y "muy
   de lejos".
5. **Foto tomada el mismo día que el pesaje**, idealmente en el mismo momento (antes o después de la manga).
6. **Luz de día**, evitando contraluz fuerte.

### Lo que NO hace falta
- **Ningún marcador, sticker ni objeto de referencia sobre el animal.** Ya lo verificamos: el sistema anda igual
  o mejor sin él.
- Ni manga, ni fondo especial, ni distancia medida con cinta.
- Vista trasera (por ahora no la usamos).

### Lo que conviene evitar
- Animales tapados por otros o por postes.
- Fotos muy oblicuas (de frente o de atrás en diagonal).
- Zoom digital.
- Que el animal esté echado o en movimiento.

---

## 4. Qué registrar de cada animal

| Dato | Obligatorio | Por qué |
|---|---|---|
| **Peso de balanza (kg)** | Sí | Es la verdad que el modelo aprende. Anotar la precisión de la balanza |
| **Identificación del animal** (caravana o número) | Sí | Imprescindible: sin esto no se puede evaluar bien el modelo |
| **Fecha y hora** | Sí | Para relacionar foto y pesaje, y seguir al animal en el tiempo |
| Raza o cruza | Sí | Barato de anotar y documenta a qué razas aplica el sistema |
| Categoría (ternero/novillo/vaquillona/vaca/toro) | Sí | Un toque en la app |
| Sexo | Sí | |
| Edad aproximada o dentición | Deseable | |
| Condición corporal, si el establecimiento la usa | Opcional | |

**Nota**: medimos que la raza y la edad, cargadas por el productor, aportan poco una vez que el modelo ve la foto
(la foto sola rinde mejor que esos datos). Se piden igual porque son gratis y permiten documentar sesgos, pero
**no van a ser obligatorios en la app**.

### Además, guardar automáticamente en cada foto
- El **archivo original sin recortar ni comprimir** (nada de reenviar por WhatsApp: destruye los metadatos).
- Los **datos EXIF** completos (modelo de teléfono, focal, resolución).
- Si se puede, los **sensores del teléfono** en el momento de la foto: inclinación y altura estimada de la cámara.
  Hoy no los usamos, pero si más adelante hace falta recuperar la escala física, sin esos datos habría que
  repetir toda la recolección.

---

## 4-bis. Cómo capturar con iPhone (paso a paso)

### Buena noticia: la inclinación ya se guarda sola
El iPhone escribe en cada foto un campo propio de Apple, `AccelerationVector`: el vector de gravedad medido por
el acelerómetro, o sea **la inclinación exacta del teléfono en el momento del disparo**. No hace falta ninguna
app especial, sale de la cámara nativa. Lo único que el iPhone **no** guarda es la **altura de la cámara sobre
el suelo**; eso lo anotamos a mano por sesión (ver abajo).

### Ajustes del iPhone antes de salir
1. **Ajustes → Cámara → Formatos → "Más compatible"**. Guarda en JPEG en vez de HEIC y evita problemas de
   conversión (el HEIC también sirve, pero complica el procesamiento).
2. **Ajustes → Cámara → Conservar ajustes → activar "Modo de cámara"**, para que no cambie solo entre tomas.
3. **Desactivar "Fotos en vivo"** (el ícono de círculos arriba): pesan el doble y no aportan.
4. **Desactivar el zoom**: usar siempre el lente principal (1x), nunca pellizcar para acercar. Si hace falta
   acercarse, caminar.
5. **GPS**: si no hay objeción del establecimiento, dejar la ubicación activada (Ajustes → Privacidad →
   Localización → Cámara → "Al usar la app"). Ayuda a identificar de qué campo salió cada lote.
6. **Ajustes → Fotos → "Descargar y conservar originales"** (no "Optimizar almacenamiento"), para que el
   original quede en el teléfono y no solo en iCloud en baja calidad.

### Cómo pasar las fotos SIN arruinarlas
Esto es lo más importante de toda la sección.

| Método | ¿Conserva la metadata? |
|---|---|
| Cable USB a la computadora (Windows: "Importar"; Mac: Captura de Imagen o Fotos) | **Sí** — el más seguro |
| AirDrop, eligiendo **"Datos originales"** al compartir | **Sí** |
| iCloud / Google Drive subiendo el **original** | Sí |
| Correo eligiendo "Tamaño real" | Sí, pero pesado |
| **WhatsApp, Telegram, Instagram** | **NO — borra todo y recomprime** |
| Google Fotos en "Ahorro de espacio" | NO |
| Convertir a PNG o captura de pantalla | **NO** |

El dataset de Mongolia que analizamos es el ejemplo perfecto del error: fueron tomadas con un iPhone 13 y nos
llegaron convertidas a PNG. **Perdieron el 100% de la metadata**, incluida la inclinación.

### Verificar en el campo, antes de sacar cientos
Sacar **2 o 3 fotos de prueba**, pasarlas a la computadora por el método elegido y correr:

```bash
python verificar_fotos.py C:\ruta\a\las\fotos
```

Avisa si la transferencia borró la metadata, si la resolución es baja o si falta la fecha. Es preferible
descubrirlo con 3 fotos que con 300. Para que además revise la inclinación hay que instalar
[exiftool](https://exiftool.org) (opcional, no bloqueante).

### Lo que sí hay que anotar a mano (una vez por sesión, no por foto)
- **Altura a la que se toman las fotos**: alcanza con "a la altura del pecho del operario" más la estatura de
  esa persona, o medir una vez con cinta. Ejemplo: *"operario de 1,75 m, foto a la altura del pecho ≈ 1,40 m"*.
- **Quién sacó las fotos** y con qué teléfono (modelo).
- Cualquier cambio de criterio a mitad de la jornada.

### Si más adelante quisieran la altura automática
Existe la opción de una app propia con ARKit, que da la posición y altura de la cámara respecto al piso en cada
disparo. Es desarrollo adicional y **no hace falta ahora**: con la inclinación que ya guarda el iPhone y la
altura anotada por sesión alcanza para reconstruir la geometría si algún día se necesita.

## 5. Errores que ya nos costaron caro (no repetirlos)

1. **Fotos sin identificación del animal.** Si no se sabe qué foto es de qué animal, no se puede separar
   entrenamiento de evaluación correctamente y los resultados quedan inflados.
2. **Peso de otra fecha.** Un animal puede variar mucho en semanas; la foto y el peso tienen que ser del mismo día.
3. **Cubrir un solo rango de peso.** El modelo no extrapola. Si solo hay novillos terminados, no va a servir para
   terneros.
4. **Pocos animales con muchas fotos.** 50 animales con 20 fotos cada uno rinde mucho menos que 300 animales con
   2 fotos.
5. **Imágenes recomprimidas o recortadas** antes de guardarlas.

---

## 6. Checklist para la visita

- [ ] ¿Cuántos animales tiene el establecimiento y en qué rangos de peso?
- [ ] ¿Cada cuánto los pesan y con qué balanza? ¿Qué precisión tiene?
- [ ] ¿Los animales están identificados individualmente (caravana)? ¿Con qué sistema?
- [ ] ¿Podemos sacar fotos el mismo día del pesaje, aprovechando la manga o el corral de encierre?
- [ ] ¿Quién saca las fotos: alguien del equipo o personal del establecimiento?
- [ ] ¿Qué teléfonos hay disponibles? (anotar modelos: conviene tener variedad, no uno solo)
- [ ] ¿Probamos la transferencia de 2–3 fotos con `verificar_fotos.py` antes de empezar en serio?
- [ ] ¿Anotamos la altura a la que se toman las fotos y quién las toma?
- [ ] ¿Podemos volver a fotografiar los mismos animales en 30–60 días?
- [ ] ¿Qué razas hay?
- [ ] ¿Hay restricciones de horario, clima o bienestar animal a considerar?
- [ ] ¿Quién es el responsable de pasarnos los datos y en qué formato?

---

## 7. Formato de entrega sugerido

Una carpeta por sesión de pesaje:

```
2026-10-15_EstablecimientoX/
├── fotos/
│   ├── 0451_1.jpg        <- caravana 0451, foto 1
│   ├── 0451_2.jpg
│   └── 0452_1.jpg
└── pesajes.csv           <- caravana, peso_kg, fecha, raza, categoria, sexo, edad, observaciones
```

Con que el nombre del archivo empiece por la caravana ya alcanza para cruzar todo automáticamente.
