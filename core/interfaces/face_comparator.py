from abc import ABC, abstractmethod
from typing import Tuple


class FaceComparator(ABC):
    """Абстракция для сравнения лиц"""

    @abstractmethod
    def compare(self, face1_path: str, face2_path: str) -> Tuple[bool, float]:
        """Сравнивает два лица и возвращает (совпадение, расстояние)"""
        pass

    @abstractmethod
    def batch_compare(self, image_paths: list) -> list:
        """Сравнивает все изображения между собой"""
        pass