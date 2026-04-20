# Modelo Exportado

El modelo final publicado en GitHub Pages es:

```text
outputs/models/gesture_recognition_(cnn_b).pt
```

<a href="./gesture_recognition_(cnn_b).pt" download>Descargar gesture_recognition_(cnn_b).pt</a>

## Metadatos

- Arquitectura: CNN-B
- Entrada esperada: imagen en escala de grises de `92x92`
- Clases: `palm`, `l`, `fist`, `fist_moved`, `thumb`, `index`, `ok`, `palm_moved`, `c`, `down`
- Framework: PyTorch

## Uso desde CLI

```powershell
.\.venv\Scripts\python.exe scripts\realtime_gesture_recognition.py --smoke-test --model "outputs\models\gesture_recognition_(cnn_b).pt"
```

```powershell
.\.venv\Scripts\python.exe scripts\realtime_gesture_recognition.py --model "outputs\models\gesture_recognition_(cnn_b).pt" --preprocess raw
```

Este modelo corresponde a la arquitectura seleccionada para las pruebas de inferencia. Su uso permite reproducir el pipeline de reconocimiento con la misma configuracion empleada en el experimento final.
