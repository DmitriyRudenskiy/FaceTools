from abc import ABC, abstractmethod
from typing import List, Tuple, Any  # Добавлен импорт Any


class FaceDetector(ABC):
    """Абстракция для детекции лиц"""

    @abstractmethod
    def detect(self, image: Any) -> List[List[float]]:
        """Детектирует лица и возвращает bounding boxes в формате [x1, y1, x2, y2]"""
        pass


class BoundingBoxProcessor(ABC):
    """Абстракция для обработки bounding box'ов"""

    @abstractmethod
    def merge_overlapping(
        self, boxes: List[List[float]], iou_threshold: float = 0.5
    ) -> List[List[float]]:
        """Объединяет пересекающиеся bounding box'ы"""
        pass

    @abstractmethod
    def calculate_square_crop(
        self, bbox: List[float], image_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Рассчитывает координаты квадратной обрезки с заданным процентом отступа"""
        pass
