# Clasificacion de Gestos de Mano con CNN

Proyecto experimental de reconocimiento de gestos de mano mediante redes neuronales convolucionales. El trabajo reproduce y adapta el flujo descrito en el paper de referencia del curso: entrenamiento de modelos CNN sobre LeapGestRecog, comparacion de arquitecturas y prueba de inferencia en tiempo real con OpenCV.

La fuente principal de resultados de este repositorio son los artefactos reproducidos en `outputs/reports`, `outputs/figures` y `outputs/models`.

## Objetivos

- Clasificar 10 gestos estaticos de mano del dataset LeapGestRecog.
- Comparar tres arquitecturas CNN con distinta profundidad y regularizacion.
- Exportar el mejor modelo para una prueba de reconocimiento en tiempo real con webcam.
- Publicar resultados, notebooks y documentacion tecnica con Sphinx y GitHub Pages.

## Dataset

Se usa LeapGestRecog, un dataset de 20,000 imagenes en escala de grises distribuidas en 10 clases balanceadas:

`palm`, `l`, `fist`, `fist_moved`, `thumb`, `index`, `ok`, `palm_moved`, `c`, `down`.

El experimento utiliza una particion estratificada 70/15/15 para entrenamiento, validacion y prueba. Las imagenes se redimensionan a `92x92` pixeles y se normalizan al rango `[0, 1]`.

## Arquitecturas Evaluadas

| Modelo | Convoluciones | Batch Normalization | Dropout |
| --- | ---: | --- | ---: |
| CNN-A | 3 | No | - |
| CNN-B | 5 | No | 0.3 |
| CNN-C | 5 | Si | 0.4 |

Los tres modelos siguen bloques convolucionales con filtros crecientes `32 -> 64 -> 128`, capas ReLU, MaxPooling y una capa densa final para clasificar las 10 clases.

## Resultados Reproducidos

| Modelo | Exactitud validacion | Exactitud prueba | F1 macro | Perdida |
| --- | ---: | ---: | ---: | ---: |
| CNN-A | 100 % | 99.967 % | 99.967 % | 0.002 |
| CNN-B | 100 % | 99.967 % | 99.967 % | 0.002 |
| CNN-C | 100 % | 99.9 % | 99.9 % | 0.006 |

El modelo exportado para inferencia es:

```text
outputs/models/gesture_recognition_(cnn_b).pt
```

Aunque los resultados sobre el conjunto estatico son casi perfectos, la prueba con webcam mostro baja estabilidad y baja confianza para varias clases. Esta diferencia sugiere una brecha de dominio entre las imagenes controladas del dataset y las condiciones reales de camara, iluminacion, fondo y posicion de la mano.

## Estructura del Proyecto

```text
.
├── notebooks/
│   ├── 01_experimento_cnn_leapgestrecog.ipynb
│   └── 02_reconocimiento_tiempo_real_webcam.ipynb
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
├── scripts/
│   └── realtime_gesture_recognition.py
├── docs/
│   └── conf.py
└── requirements.txt
```

## Instalacion

Crear y activar un entorno virtual, luego instalar dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para construir la documentacion:

```powershell
python -m pip install -r docs\requirements.txt
sphinx-build -b html docs docs\_build\html
```

## Uso

Ejecutar los notebooks en orden:

1. `notebooks/01_experimento_cnn_leapgestrecog.ipynb`: descarga, preparacion, entrenamiento, evaluacion y exportacion.
2. `notebooks/02_reconocimiento_tiempo_real_webcam.ipynb`: carga del modelo exportado y prueba del pipeline de webcam.

Smoke test sin abrir camara:

```powershell
.\.venv\Scripts\python.exe scripts\realtime_gesture_recognition.py --smoke-test --model "outputs\models\gesture_recognition_(cnn_b).pt"
```

Ejecucion con webcam:

```powershell
.\.venv\Scripts\python.exe scripts\realtime_gesture_recognition.py --model "outputs\models\gesture_recognition_(cnn_b).pt" --preprocess raw
```

## Documentacion

La documentacion Sphinx esta preparada para GitHub Pages con el tema Furo. Incluye:

- contenido renderizado de notebooks sin reejecutarlos;
- tablas de resultados desde `outputs/reports`;
- figuras desde `outputs/figures`;
- API del script de inferencia;
- enlace al modelo final exportado.

Construccion local:

```powershell
sphinx-build -b html -W --keep-going docs docs\_build\html
```

El workflow `.github/workflows/pages.yml` construye y publica el HTML automaticamente en GitHub Pages.
