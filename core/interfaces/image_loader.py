from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class ImageLoader(ABC):
    """Абстракция для загрузки изображений из директории."""

    @abstractmethod
    def load_images(self, directory: str) -> List[Tuple[str, np.ndarray]]:
        """Загружает изображения из указанной директории.

        Args:
            directory: Путь к директории с изображениями

        Returns:
            Список кортежей (путь_к_файлу, изображение_в_формате_ndarray)
        """
        pass

    @abstractmethod
    def validate_directory(self, directory: str) -> bool:
        """Проверяет существование и доступность директории.

        Args:
            directory: Путь к директории

        Returns:
            True, если директория существует и доступна для чтения
        """
        pass