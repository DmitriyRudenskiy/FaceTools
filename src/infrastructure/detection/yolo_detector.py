"""
Модуль детектора и компаратора лиц на основе YOLO.
Реализует интерфейс FaceDetector для обнаружения лиц с использованием YOLOv8.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from ultralytics import YOLO  # type: ignore
from src.core.interfaces import FaceDetector
from src.core.exceptions import FaceDetectionError



class YOLOFaceDetector(FaceDetector):
    """Детекция и сравнение лиц с использованием YOLO"""

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

    def extract_embeddings(self, image: Any) -> List[np.ndarray]:
        """
        Извлекает эмбеддинги для всех обнаруженных лиц на изображении.

        Args:
            image: Входное изображение для обработки

        Returns:
            Список эмбеддингов для каждого обнаруженного лица
        """
        try:
            results = self.model.predict(image, verbose=False)[0]
            embeddings = []
            for box in results.boxes:
                # Извлекаем эмбеддинг из результатов детекции
                embedding = box.embeddings[0].cpu().numpy() if hasattr(box, 'embeddings') else None
                if embedding is not None:
                    embeddings.append(embedding)
            return embeddings
        except Exception as e:
            raise FaceDetectionError(f"Ошибка извлечения эмбеддингов: {str(e)}") from e

    def compare_faces(self,
                     image1: Any,
                     image2: Any,
                     threshold: float = 0.6) -> Tuple[bool, float]:
        """
        Сравнивает два лица на разных изображениях.

        Args:
            image1: Первое изображение с лицом
            image2: Второе изображение с лицом
            threshold: Пороговое значение косинусной схожести (по умолчанию 0.6)

        Returns:
            Кортеж (bool, float): Результат сравнения и значение схожести
        """
        try:
            # Извлекаем эмбеддинги для обоих изображений
            emb1_list = self.extract_embeddings(image1)
            emb2_list = self.extract_embeddings(image2)

            if not emb1_list or not emb2_list:
                raise FaceDetectionError("Не удалось извлечь эмбеддинги для сравнения")

            # Берем первые эмбеддинги из каждого списка
            emb1 = emb1_list[0]
            emb2 = emb2_list[0]

            # Вычисляем косинусную схожесть
            similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]

            # Нормализуем схожесть в диапазон [0, 1]
            normalized_similarity = (similarity + 1) / 2

            return (normalized_similarity >= threshold, normalized_similarity)

        except Exception as e:
            raise FaceDetectionError(f"Ошибка сравнения лиц: {str(e)}") from e


class DefaultBoundingBoxProcessor:
    """Простая реализация обработчика bounding box'ов по умолчанию."""

    def merge_overlapping(self, boxes: List[List[float]], iou_threshold: float = 0.5) -> List[List[float]]:
        """
        Заглушка для функции слияния пересекающихся боксов.
        В текущей реализации просто возвращает исходные боксы.
        """
        # TODO: Реализовать алгоритм Non-Maximum Suppression (NMS)
        return boxes

    def calculate_square_crop(
            self, bbox: List[float], image_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Рассчитывает координаты квадратной обрезки на основе bounding box.
        Делегирует расчет вспомогательному классу SquareCropCalculator.
        """
        from src.utils.image_utils import SquareCropCalculator
        calculator = SquareCropCalculator()
        x1, y1, x2, y2 = bbox
        # SquareCropCalculator ожидает целочисленные координаты
        crop_coords = calculator.calculate_crop(
            (int(x1), int(y1), int(x2), int(y2)),
            image_size[0],
            image_size[1]
        )
        return crop_coords