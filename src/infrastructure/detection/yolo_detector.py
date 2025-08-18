# src/infrastructure/detection/yolo_detector.py

import os
from pathlib import Path
from typing import Any, List, Tuple

from ultralytics import YOLO

from src.core.interfaces import (  # Импорт из core.interfaces
    BoundingBoxProcessor, FaceDetector)
from src.domain.face import BoundingBox


class YOLOFaceDetector(FaceDetector):
    """Детекция лиц с использованием YOLO"""

    def __init__(self, model_path: str = None):
        # --- Способ 1a: Использовать путь относительно корня проекта ---
        if model_path is None:
            # Предполагаем, что модель лежит в <корень_проекта>/models/yolov8n-face.pt
            # __file__ = .../FaceTools/src/infrastructure/detection/yolo_detector.py
            # parents[0] = detection
            # parents[1] = infrastructure
            # parents[2] = src
            # parents[3] = FaceTools (корень проекта)
            project_root = (
                Path(__file__).resolve().parents[3]
            )  # src/infrastructure/detection -> project_root
            model_path = project_root / "models" / "yolov8n-face.pt"
        model_path = str(model_path)  # Преобразуем Path в строку

        try:
            # Проверяем существование файла модели
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Файл модели не найден: {model_path}")
            self.model = YOLO(model_path)
        except Exception as e:
            from src.core.exceptions import FaceDetectionError

            raise FaceDetectionError(
                f"Ошибка инициализации модели YOLO: {str(e)}"
            ) from e

    def detect(self, image: Any) -> List[BoundingBox]:
        try:
            results = self.model.predict(image.data, verbose=False)[0]
            boxes = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append(BoundingBox(x1, y1, x2, y2))
            return boxes
        except Exception as e:
            from src.core.exceptions import FaceDetectionError

            raise FaceDetectionError(f"Ошибка детекции лиц: {str(e)}") from e


class DefaultBoundingBoxProcessor(BoundingBoxProcessor):
    """Обработка bounding box'ов"""

    def calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        x1_inter = max(box1.x1, box2.x1)
        y1_inter = max(box1.y1, box2.y1)
        x2_inter = min(box1.x2, box2.x2)
        y2_inter = min(box1.y2, box2.y2)
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        return inter_area / (area1 + area2 - inter_area) if (area1 + area2) > 0 else 0

    def merge_overlapping(
        self, boxes: List[BoundingBox], iou_threshold: float = 0.5
    ) -> List[BoundingBox]:
        if not boxes:
            return []
        # Сортируем по площади (от большего к меньшему)
        boxes_sorted = sorted(boxes, key=lambda b: b.width * b.height, reverse=True)
        merged = []
        for box in boxes_sorted:
            is_merged = False
            for i, merged_box in enumerate(merged):
                if self.calculate_iou(box, merged_box) > iou_threshold:
                    # Объединяем bounding box'ы
                    x1 = min(box.x1, merged_box.x1)
                    y1 = min(box.y1, merged_box.y1)
                    x2 = max(box.x2, merged_box.x2)
                    y2 = max(box.y2, merged_box.y2)
                    merged[i] = BoundingBox(x1, y1, x2, y2)
                    is_merged = True
                    break
            if not is_merged:
                merged.append(box)
        return merged

    def calculate_square_crop(
        self, bbox: BoundingBox, image_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Рассчитывает квадратную область для обрезки лица"""
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        img_w, img_h = image_size
        # Вычисляем центр и размер
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        size = max(bbox.width, bbox.height) * 1.25  # 25% отступа
        # Координаты квадрата
        half_size = size / 2
        new_x1 = max(0, center_x - half_size)
        new_y1 = max(0, center_y - half_size)
        new_x2 = min(img_w, center_x + half_size)
        new_y2 = min(img_h, center_y + half_size)
        # Коррекция размера до квадрата
        actual_size = min(new_x2 - new_x1, new_y2 - new_y1)
        if new_x2 - new_x1 > actual_size:
            new_x1 = center_x - actual_size / 2
            new_x2 = new_x1 + actual_size
        if new_y2 - new_y1 > actual_size:
            new_y1 = center_y - actual_size / 2
            new_y2 = new_y1 + actual_size
        return (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
