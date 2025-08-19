"""
Модуль детектора лиц на основе YOLO.
Реализует интерфейс FaceDetector для обнаружения лиц с использованием YOLOv8.
"""

from pathlib import Path
from typing import Any, List, Optional

from ultralytics import YOLO  # type: ignore
from src.core.interfaces import FaceDetector
from src.core.exceptions import FaceDetectionError


class YOLOFaceDetector(FaceDetector):
    """Детекция лиц с использованием YOLO"""

    def __init__(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            project_root = Path(__file__).resolve().parents[3]
            model_path = str(project_root / "models" / "yolov8n-face.pt")

        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Файл модели не найден: {model_path}")
            self.model = YOLO(model_path)
        except Exception as e:
            raise FaceDetectionError(
                f"Ошибка инициализации модели YOLO: {str(e)}"
            ) from e

    def detect(self, image: Any) -> List[List[float]]:
        try:
            results = self.model.predict(image, verbose=False)[0]
            boxes: List[List[float]] = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append([x1, y1, x2, y2])
            return boxes
        except Exception as e:
            raise FaceDetectionError(f"Ошибка детекции лиц: {str(e)}") from e
