from abc import ABC, abstractmethod
from typing import Tuple, List


class FaceComparator(ABC):
    """Абстракция для сравнения лиц"""

    @abstractmethod
    def compare(self, face1_path: str, face2_path: str) -> Tuple[bool, float]:
        """Сравнивает два лица и возвращает (совпадение, расстояние)"""
        pass

    @abstractmethod
    def batch_compare(self, image_paths: List[str]) -> List[List[Tuple[bool, float]]]:
        """Сравнивает все изображения между собой и возвращает матрицу сравнения"""
        pass
