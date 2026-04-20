# Resultados

## Comparacion de Modelos

```{csv-table}
:file: data/resultados_modelos_formateados.csv
:header-rows: 1
```

Los tres modelos alcanzan un desempeno alto en el conjunto de prueba. CNN-B conserva el mismo valor de exactitud y F1 macro que CNN-A, con la menor perdida observada entre ambas arquitecturas.

## Curvas de Aprendizaje

```{image} ../outputs/figures/curvas_aprendizaje_modelos.png
:alt: Curvas de aprendizaje de los modelos CNN
:width: 100%
```

## Matriz de Confusion

```{image} ../outputs/figures/matriz_confusion_cnn_b.png
:alt: Matriz de confusion del modelo CNN-B
:width: 100%
```

## Metricas por Clase

```{csv-table}
:file: data/metricas_por_clase_formateadas.csv
:header-rows: 1
```

Las metricas por clase muestran un comportamiento uniforme en el conjunto de prueba. La unica variacion apreciable aparece en las clases `palm` y `fist`, donde se registra una diferencia pequena entre precision y recall.

## Tiempo Real

```{csv-table}
:file: ../outputs/reports/cuadro_v_tiempo_real_pendiente.csv
:header-rows: 1
```

La evaluacion con webcam evidencia que el desempeno sobre imagenes estaticas no se traslada automaticamente a un entorno de captura real. Las predicciones fueron sensibles a iluminacion, fondo, ruido de camara y posicion de la mano.
