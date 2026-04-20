# Clasificación de Gestos de Mano con CNN

Esta documentación acompaña el experimento de clasificación de gestos de mano con redes neuronales convolucionales. El sitio presenta los notebooks, resultados reproducidos, figuras, modelo final y pipeline de inferencia en tiempo real.

```{toctree}
:maxdepth: 2
:caption: Contenido

experimento
resultados
notebooks/01_experimento_cnn_leapgestrecog
notebooks/02_reconocimiento_tiempo_real_webcam
modelo
api
```

## Material del experimento

- Notebook experimental: `notebooks/01_experimento_cnn_leapgestrecog.ipynb`
- Notebook de webcam: `notebooks/02_reconocimiento_tiempo_real_webcam.ipynb`
- Modelo final: `outputs/models/gesture_recognition_(cnn_b).pt`
- Reportes: `outputs/reports`
- Figuras: `outputs/figures`

## Resultado principal

El modelo CNN-B alcanza una exactitud de prueba de `99.967 %` y un F1 macro de `99.967 %` sobre el conjunto estático. En webcam, el rendimiento fue menos estable debido a la diferencia entre las condiciones controladas del dataset y el entorno real de captura.
