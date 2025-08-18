from abc import ABC, abstractmethod
from typing import Any


class ImageLoader(ABC):
    """Абстракция для загрузки изображений"""

    @abstractmethod
    def load(self, path: str) -> Any:
        """Загружает изображение из файла"""
        pass
