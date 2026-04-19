from __future__ import annotations

import argparse
import csv
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


DEFAULT_IMAGE_SIZE = (92, 92)
MODEL_CONFIGS = {
    "CNN_A": {
        "dropout": 0.0,
        "blocks": [
            {"out_channels": 32, "convs": 1, "use_bn": False},
            {"out_channels": 64, "convs": 1, "use_bn": False},
            {"out_channels": 128, "convs": 1, "use_bn": False},
        ],
    },
    "CNN_B": {
        "dropout": 0.3,
        "blocks": [
            {"out_channels": 32, "convs": 2, "use_bn": False},
            {"out_channels": 64, "convs": 2, "use_bn": False},
            {"out_channels": 128, "convs": 1, "use_bn": False},
        ],
    },
    "CNN_C": {
        "dropout": 0.4,
        "blocks": [
            {"out_channels": 32, "convs": 2, "use_bn": True},
            {"out_channels": 64, "convs": 2, "use_bn": True},
            {"out_channels": 128, "convs": 1, "use_bn": True},
        ],
    },
}


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, convs: int, use_bn: bool):
        super().__init__()

        layers = []
        for i in range(convs):
            current_in = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(current_in, out_channels, kernel_size=3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GestureCNN(nn.Module):
    def __init__(self, model_key: str, num_classes: int, image_size: tuple[int, int]):
        super().__init__()

        normalized_key = normalize_model_key(model_key)
        if normalized_key not in MODEL_CONFIGS:
            raise ValueError(f"Modelo desconocido: {model_key}")

        config = MODEL_CONFIGS[normalized_key]
        self.model_key = normalized_key
        self.image_size = image_size

        self.features = self._build_features(config["blocks"])
        flattened = self._get_flattened_size()

        classifier_layers = [
            nn.Flatten(),
            nn.Linear(flattened, 256),
            nn.ReLU(inplace=True),
        ]

        if config["dropout"] > 0:
            classifier_layers.append(nn.Dropout(config["dropout"]))

        classifier_layers.append(nn.Linear(256, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    @staticmethod
    def _build_features(blocks: list[dict]) -> nn.Sequential:
        layers = []
        in_channels = 1

        for block in blocks:
            layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=block["out_channels"],
                    convs=block["convs"],
                    use_bn=block["use_bn"],
                )
            )
            in_channels = block["out_channels"]

        return nn.Sequential(*layers)

    def _get_flattened_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.image_size[1], self.image_size[0])
            return self.features(dummy).view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def normalize_model_key(model_key: str) -> str:
    return model_key.strip().upper().replace("-", "_")


def display_model_key(model_key: str) -> str:
    return normalize_model_key(model_key).replace("_", "-")


def parse_image_size(value) -> tuple[int, int]:
    if value is None:
        return DEFAULT_IMAGE_SIZE
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"image_size invalido en checkpoint: {value!r}")


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def checkpoint_sort_key(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def best_model_from_report(project_root: Path) -> Path | None:
    report_path = project_root / "outputs" / "reports" / "resultados_modelos_obtenidos.csv"
    if not report_path.exists():
        return None

    with report_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return None

    def row_key(row: dict[str, str]) -> tuple[float, float, float]:
        return (
            float(row.get("F1-macro", 0.0)),
            float(row.get("Acc. Test", 0.0)),
            -float(row.get("Loss", 999999.0)),
        )

    best_row = max(rows, key=row_key)
    model_key = normalize_model_key(best_row["Modelo"]).lower().replace("_", "-")
    candidate = project_root / "outputs" / "models" / f"{model_key}_best.pt"
    return candidate if candidate.exists() else None


def choose_model_path(project_root: Path, explicit_model: str | None) -> Path:
    if explicit_model:
        path = Path(explicit_model)
        if not path.is_absolute():
            path = project_root / path
        if not path.exists():
            raise FileNotFoundError(f"No existe el modelo indicado: {path}")
        return path

    models_dir = project_root / "outputs" / "models"
    exported = sorted(models_dir.glob("gesture_recognition_(*).pt"), key=checkpoint_sort_key, reverse=True)
    if exported:
        return exported[0]

    report_model = best_model_from_report(project_root)
    if report_model is not None:
        return report_model

    raise FileNotFoundError(
        "No encontre un checkpoint exportado en outputs/models/gesture_recognition_(*).pt "
        "ni un *_best.pt resoluble desde outputs/reports/resultados_modelos_obtenidos.csv."
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidio --device cuda, pero PyTorch no detecta CUDA.")
    return torch.device(requested)


def load_torch_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError(f"El checkpoint no es un dict: {path}")
    return checkpoint


def load_checkpoint_model(path: Path, device: torch.device) -> tuple[GestureCNN, dict]:
    checkpoint = load_torch_checkpoint(path, device)
    required = {"model_key", "class_names", "state_dict"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint incompleto. Faltan campos: {sorted(missing)}")

    class_names = list(checkpoint["class_names"])
    image_size = parse_image_size(checkpoint.get("image_size"))
    model_key = normalize_model_key(str(checkpoint["model_key"]))

    model = GestureCNN(model_key=model_key, num_classes=len(class_names), image_size=image_size)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    metadata = {
        "model_key": model_key,
        "class_names": class_names,
        "image_size": image_size,
        "best_val_accuracy": checkpoint.get("best_val_accuracy"),
    }
    return model, metadata


def skin_mask_from_bgr(roi_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    ycrcb_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 180, 135]))
    hsv_mask = cv2.inRange(hsv, np.array([0, 20, 40]), np.array([35, 255, 255]))
    mask = cv2.bitwise_or(ycrcb_mask, hsv_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def bright_foreground_fallback(gray: np.ndarray) -> np.ndarray:
    threshold = np.percentile(gray, 65)
    mask = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def darken_background(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    mask = skin_mask_from_bgr(roi_bgr)
    mask_ratio = cv2.countNonZero(mask) / mask.size
    if mask_ratio < 0.03 or mask_ratio > 0.85:
        mask = bright_foreground_fallback(enhanced)

    feather = cv2.GaussianBlur(mask, (21, 21), 0).astype(np.float32) / 255.0
    foreground = np.clip(enhanced.astype(np.float32) * 1.35 + 20.0, 0, 255)
    background = enhanced.astype(np.float32) * 0.12
    darkened = foreground * feather + background * (1.0 - feather)
    return np.clip(darkened, 0, 255).astype(np.uint8)


def preprocess_roi(roi_bgr: np.ndarray, image_size: tuple[int, int], mode: str) -> tuple[torch.Tensor, np.ndarray]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    if mode == "otsu":
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, gray = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "dark-bg":
        gray = darken_background(roi_bgr)
    elif mode != "raw":
        raise ValueError(f"Modo de preprocesamiento desconocido: {mode}")

    resized = cv2.resize(gray, image_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor, resized


def center_roi(frame_shape: tuple[int, int, int], roi_size: int) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    size = max(32, min(roi_size, width - 1, height - 1))
    x = (width - size) // 2
    y = (height - size) // 2
    return x, y, size, size


def predict_probabilities(model: nn.Module, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.inference_mode():
        logits = model(tensor.to(device))
        probabilities = torch.softmax(logits, dim=1)[0]
    return probabilities.detach().cpu().numpy()


def draw_text(frame: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.7) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def draw_preview(frame: np.ndarray, preview_gray: np.ndarray) -> None:
    preview = cv2.resize(preview_gray, (120, 120), interpolation=cv2.INTER_NEAREST)
    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    top = 12
    left = frame.shape[1] - preview_bgr.shape[1] - 12
    frame[top : top + preview_bgr.shape[0], left : left + preview_bgr.shape[1]] = preview_bgr
    cv2.rectangle(frame, (left, top), (left + 120, top + 120), (255, 255, 255), 1)


def run_smoke_test(args: argparse.Namespace) -> None:
    project_root = project_root_from_script()
    device = resolve_device(args.device)
    model_path = choose_model_path(project_root, args.model)
    model, metadata = load_checkpoint_model(model_path, device)

    fake_frame = np.zeros((args.roi_size, args.roi_size, 3), dtype=np.uint8)
    tensor, _ = preprocess_roi(fake_frame, metadata["image_size"], args.preprocess)
    probabilities = predict_probabilities(model, tensor, device)

    print(f"Modelo: {model_path}")
    print(f"Arquitectura: {display_model_key(metadata['model_key'])}")
    print(f"Clases: {metadata['class_names']}")
    print(f"Image size: {metadata['image_size']}")
    print(f"Preprocesamiento: {args.preprocess}")
    print(f"Probabilidades: shape={probabilities.shape}, suma={probabilities.sum():.6f}")

    if len(probabilities) != len(metadata["class_names"]):
        raise RuntimeError("La salida no coincide con la cantidad de clases.")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-4):
        raise RuntimeError("Las probabilidades no suman aproximadamente 1.")


def run_webcam(args: argparse.Namespace) -> None:
    project_root = project_root_from_script()
    device = resolve_device(args.device)
    model_path = choose_model_path(project_root, args.model)
    model, metadata = load_checkpoint_model(model_path, device)

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.camera, backend)
    if not camera.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara {args.camera}.")

    history: deque[np.ndarray] = deque(maxlen=max(1, args.smooth_window))
    fps = 0.0
    last_time = time.perf_counter()
    window_name = "Reconocimiento de gestos - webcam"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"Modelo: {model_path}")
    print(f"Arquitectura: {display_model_key(metadata['model_key'])}")
    print(f"Clases: {metadata['class_names']}")
    print(f"Image size: {metadata['image_size']}")
    print(f"Dispositivo: {device}")
    print("Coloca la mano dentro del cuadro. Salir con q o Esc.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("No se pudo leer un frame de la camara.")

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            x, y, w, h = center_roi(frame.shape, args.roi_size)
            roi = frame[y : y + h, x : x + w]
            tensor, preview = preprocess_roi(roi, metadata["image_size"], args.preprocess)
            probabilities = predict_probabilities(model, tensor, device)
            history.append(probabilities)

            smooth_probabilities = np.mean(np.stack(history), axis=0)
            class_index = int(np.argmax(smooth_probabilities))
            confidence = float(smooth_probabilities[class_index])
            predicted_label = metadata["class_names"][class_index]
            display_label = predicted_label if confidence >= args.confidence_threshold else "incierto"

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_time, 1e-6)
            fps = instant_fps if fps == 0.0 else (0.9 * fps + 0.1 * instant_fps)
            last_time = now

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
            draw_text(frame, f"{display_label} | conf: {confidence:.2f}", (20, 36), 0.8)
            draw_text(frame, f"prep: {args.preprocess} | fps: {fps:.1f} | device: {device}", (20, 70), 0.6)
            draw_text(frame, f"model: {display_model_key(metadata['model_key'])}", (20, 102), 0.6)
            draw_preview(frame, preview)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconocimiento de gestos de mano en tiempo real con OpenCV y PyTorch."
    )
    parser.add_argument("--model", type=str, default=None, help="Ruta al checkpoint .pt.")
    parser.add_argument("--camera", type=int, default=0, help="Indice de la camara web.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Dispositivo de inferencia.",
    )
    parser.add_argument(
        "--preprocess",
        choices=["dark-bg", "raw", "otsu"],
        default="raw",
        help="raw replica el entrenamiento; otsu binariza la ROI; dark-bg realza la mano y oscurece el fondo.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.60, help="Umbral para mostrar una clase.")
    parser.add_argument("--smooth-window", type=int, default=7, help="Cantidad de frames para suavizar.")
    parser.add_argument("--roi-size", type=int, default=260, help="Tamano del cuadro ROI en pixeles.")
    parser.add_argument("--no-mirror", action="store_true", help="No invertir horizontalmente la webcam.")
    parser.add_argument("--smoke-test", action="store_true", help="Cargar el modelo y probar una inferencia sin camara.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(args)
    else:
        run_webcam(args)


if __name__ == "__main__":
    main()
