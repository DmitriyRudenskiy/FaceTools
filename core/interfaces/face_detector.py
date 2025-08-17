from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class FaceDetector(ABC):
    """Абстракция для детекции лиц на изображениях."""

    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Обнаруживает лица на изображении.

        Args:
            image: Изображение в формате numpy array

        Returns:
            Список словарей с информацией о найденных лицах
            (bbox, landmarks, confidence и т.д.)
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """Возвращает информацию о модели детекции.

        Returns:
            Словарь с информацией о модели
        """
        pass