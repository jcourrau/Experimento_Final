
# Notas

## IV. DISEÑO EXPERIMENTAL

### IV-A. Dataset
* 20mil imagenes -> ✔
* 10 categorias -> ✔

### IV-B. Preprocesamiento y Aumento de Datos
* Redimensionadas a 64 x 64 -> 96x96
* Transformaciones -> ✔


### IV-C. Arquitecturas Evaluadas
???
* *Cuadro II*
* *Graficos*


### IV-E Entrenamiento
* TensorFlow/Keras -> Pytorch
* 30 épocas -> ✔
* RTX 3060 -> ✔
* Opt Adam -> ✔
* Epocas 30 -> ✔
* ReduceLROnPlateau (factor=0.5, patience=3,)  -> ✔

### IV-F. Pipeline de Reconocimiento en Tiempo Real
* Min 24 FPS -> ?
* Open CV 4.8 -> ?
* 64 x 64 -> 92 x 92
* Otsu (umbralización adaptativa) -> Se aleja a las imagenes de entrenamiento, descartado. 

## IV. DISEÑO EXPERIMENTAL


### V-A. Desempeño de Clasificación (RQ1 y RQ2)
* *Cuadro III*
* Mejor modelo: CNN-C -> CNN-B
* *Rehacer conclusiones basadas en resultados*

### V-B. Desempeño por Clase
* *Cuadro IV*
* *Rehacer conclusiones basadas en resultados*

### V-C. Curvas de Aprendizaje
* *Grafico de curvas*
* *Rehacer conclusiones basadas en resultados*

### V-D. Reconocimiento en Tiempo Real (RQ3)
* *Cuadro V*
* *Rehacer conclusiones basadas en resultados*


### V-E. Análisis de Errores
* *Rehacer conclusiones basadas en resultados*