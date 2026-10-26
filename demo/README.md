# Sets de fotos para la demo

**Son para mostrar el modelo, no para medirlo.** Dos de los tres están elegidos a propósito (por resultado o porque el modelo
entrenó con esas vacas). Si en la presentación se muestra un número de acá, hay que decir cuál de los tres es.

| Carpeta | Modelo | Fotos | Cómo se eligieron | Qué es en honestidad |
|---|---|---|---|---|
| `demo_bangladesh_v1/` | **v1** (`modelos/`) | 20 | Animales del split de **test** de Bangladesh (el v1 no los vio), uno por animal, repartidos por peso (82–268 kg). No elegidas por resultado | El mejor caso del v1: su terreno. Medido en toda la tanda: 15,6 %. |
| `demo_propio_v2/` | **v2** (`modelos_v2/`) | 15 | Del 8/10 y 22/10, una por animal, laterales completas, repartidas por peso (292–485 kg) | **El v2 se entrenó con estas vacas.** Muestra el modelo en su terreno, no una prueba. |
| `demo_6-10_mejores_v2/` | **v2** | 12 | Las **12 con menor error** del v2 sobre el 6/10 (una por animal) | **Elegidas por su resultado.** Aun así el v2 solo acierta ~18 % en promedio sobre ellas. |

Cada carpeta tiene las fotos y un `pesos.csv` (`foto, peso_kg, animal`).

## Cómo usarlos

Desde `estimador/`:

```bash
# página web: arrastrar las fotos de la carpeta y cargar el peso real de pesos.csv
.venv\Scripts\python probar.py --web

# o por consola, con el modelo que corresponde
.venv\Scripts\python probar.py --carpeta ..\demo\demo_bangladesh_v1  --pesos ..\demo\demo_bangladesh_v1\pesos.csv  --modelo v1=modelos
.venv\Scripts\python probar.py --carpeta ..\demo\demo_propio_v2      --pesos ..\demo\demo_propio_v2\pesos.csv      --modelo v2=modelos_v2
.venv\Scripts\python probar.py --carpeta ..\demo\demo_6-10_mejores_v2 --pesos ..\demo\demo_6-10_mejores_v2\pesos.csv --modelo v2=modelos_v2
```

## Qué esperar (corrida de verificación, una vez por set)

| Set | Error medio (MAPE) | Predecir siempre el peso medio de la muestra | "Confiable" |
|---|---|---|---|
| `demo_bangladesh_v1` con v1 | 12,6 % | 22,9 % | 11 de 20 |
| `demo_propio_v2` con v2 | 7,7 % | 9,6 % | 15 de 15 |
| `demo_6-10_mejores_v2` con v2 | 17,8 % | 7,5 % | 4 de 12 |

La referencia "predecir siempre el peso medio" usa la mediana de la propia muestra, que ya conoce las respuestas; en el set
del 6/10 queda por debajo del modelo **incluso después de elegir sus mejores fotos**. Eso es lo que hay que decir si preguntan.

Armados con `bovinovision/53_armar_demos.py` (se regeneran con el mismo comando).
