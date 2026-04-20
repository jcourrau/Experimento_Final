# Diseno Experimental

## Dataset

El experimento usa LeapGestRecog, un conjunto de 20,000 imagenes de gestos de mano en escala de grises. Las clases son balanceadas y representan 10 gestos: `palm`, `l`, `fist`, `fist_moved`, `thumb`, `index`, `ok`, `palm_moved`, `c` y `down`.

Las imagenes se dividen con una particion estratificada 70/15/15 para entrenamiento, validacion y prueba. El pipeline del notebook redimensiona cada imagen a `92x92` pixeles y normaliza los valores al rango `[0, 1]`.

## Aumento de Datos

```{csv-table}
:file: ../outputs/reports/cuadro_i_aumento_datos.csv
:header-rows: 1
```

## Arquitecturas

```{csv-table}
:file: ../outputs/reports/cuadro_ii_arquitecturas.csv
:header-rows: 1
```

```{image} ../notebooks/cnn_architectures_horizontal_titled.png
:alt: Arquitecturas CNN evaluadas
:width: 100%
```

## Entrenamiento

Los modelos se entrenan en PyTorch durante 30 epocas con el optimizador Adam, perdida de entropia cruzada y un planificador de tasa de aprendizaje `ReduceLROnPlateau`. El proceso registra los pesos del mejor modelo y los historiales de entrenamiento para analizar convergencia, perdida y exactitud.

## Reconocimiento en Tiempo Real

El pipeline de webcam usa OpenCV para capturar frames, extraer una region de interes centrada, aplicar preprocesamiento y ejecutar inferencia con el modelo exportado. La prueba real fue inestable para varias clases, lo que indica una brecha de dominio entre las imagenes controladas del dataset y las condiciones reales de iluminacion, fondo y camara.
