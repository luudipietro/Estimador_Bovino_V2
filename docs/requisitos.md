# Requisitos que cubre el servicio de estimación

> Extracto del **TP8 (Plan de Testing, §3 y §4)**. La fuente de verdad es la **ERS del TP7 (IEEE 830)**, que está en
> SVN. Si el texto de la ERS difiere, manda la ERS: reemplazar las redacciones de abajo y avisar.

El backend C# cubre el resto de los requisitos (RF-10 a RF-18 en el frontend; RF-21 y RF-23 a RF-35 en el backend).
Este servicio cubre lo siguiente:

| Requisito | Qué exige | Cómo se cumple en el servicio | Cómo se prueba |
|---|---|---|---|
| **RF-15** (parcial) | Validación automática de la calidad de la imagen | Validación de entrada (formato, corrupción, resolución) y motivos de calidad del modelo (R-24, R-25) | Unitarias + fixtures de imágenes |
| **RF-19** | Envío asincrónico de la imagen al servicio de procesamiento | `POST /v1/estimaciones` responde 202 y deja un trabajo | API |
| **RF-20** | Estimación del peso con modelos de visión artificial | Caso de uso `ProcesarEstimacion` + adaptador del modelo | Unitarias (fake) + integración (`-m modelo`) |
| **RF-21** (soporte) | Notificación al finalizar la estimación | Callback firmado al backend C#. **La notificación al usuario la emite el backend.** | API con `respx` |
| **RF-22** | Notificación y manejo de errores durante la estimación | Catálogo de errores estable y estados `completado` / `fallido` (ver `errores.md`) | Unitarias + API |
| **RNF-01** | Tiempo total de estimación ≤ 60 s | Presupuesto por trabajo, inferencia en hilo y medición por etapa | `scripts/carga_concurrente.py` |
| **RNF-02** | MAPE < 20 % | Evaluación por animal sobre ≥ 100 registros con peso de báscula | `scripts/evaluar_modelo.py` |
| **RNF-03** (soporte) | Seguimiento simultáneo de ≥ 50 animales | 50 trabajos concurrentes dentro de RNF-01 | `scripts/carga_concurrente.py --n 50` |

## Estado del modelo (leer antes de probar RNF-02)

- Entrenado con 4.540 fotos de Bangladesh. **15,6 % de MAPE** en ese dominio (IC 95 %: 14,0–16,3).
- **No extrapola**: nunca predice por encima de ~270 kg.
- Sobre animales de otro rango falla: en Mongolia (341–644 kg) da **61 % de error**. El dataset de Mendeley que el
  TP8 propone como complemento es de ese tipo de animales, así que **con el modelo actual RNF-02 no va a pasar sobre
  esos datos**. Es esperado y está medido; no es un bug ni se arregla con umbrales.
- RNF-02 se cumple con el **modelo refinado con el dataset propio** (300–400 animales, ver `reglas-dominio.md`,
  R-14 a R-17). Mientras tanto `evaluar_modelo.py` informa el número real, por banda, con el aviso de que el
  modelo no cubre el rango.
- La meta del TP8 (**MAPE < 20 %**) es menos exigente que la del Acta (< 15 %). El criterio de aprobación es el del
  TP8, pero se informa también contra el Acta.

## Nota sobre "proporciones"

El TP8 dice que el servicio "calcula proporciones" entre puntos anatómicos. En el modelo vigente, **los keypoints
se usan para aislar y recortar al animal y para las reglas de "no sé"**; el peso lo predice la CNN sobre el
recorte (la geometría dejó de aportar, ver `bovinovision/CLAUDE.md`). Las pruebas unitarias de "lógica de ratios"
del TP8 §2.2 se aplican a esa lógica de recorte y validación. Si se exponen ratios en la respuesta, son
**informativos** y no deben presentarse como la base de la estimación.
