# Resultados

## Comparación de Modelos

```{csv-table}
:file: data/resultados_modelos_formateados.csv
:header-rows: 1
```

Los tres modelos alcanzan un desempeño alto en el conjunto de prueba. CNN-B conserva el mismo valor de exactitud y F1 macro que CNN-A, con la menor pérdida observada entre ambas arquitecturas.

## Curvas de Aprendizaje

```{image} ../outputs/figures/curvas_aprendizaje_modelos.png
:alt: Curvas de aprendizaje de los modelos CNN
:width: 100%
```

## Matriz de Confusión

```{image} ../outputs/figures/matriz_confusion_cnn_b.png
:alt: Matriz de confusión del modelo CNN-B
:width: 100%
```

## Métricas por Clase

```{csv-table}
:file: data/metricas_por_clase_formateadas.csv
:header-rows: 1
```

Las métricas por clase muestran un comportamiento uniforme en el conjunto de prueba. La única variación apreciable aparece en las clases `palm` y `fist`, donde se registra una diferencia pequeña entre precisión y recall.

## Tiempo Real

```{csv-table}
:file: ../outputs/reports/cuadro_v_tiempo_real_pendiente.csv
:header-rows: 1
```

La evaluación con webcam evidencia que el desempeño sobre imágenes estáticas no se traslada automáticamente a un entorno de captura real. Las predicciones fueron sensibles a iluminación, fondo, ruido de cámara y posición de la mano.
