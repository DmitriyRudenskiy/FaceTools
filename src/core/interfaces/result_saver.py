from abc import ABC, abstractmethod
from typing import Any  # Добавлен импорт Any


class ResultSaver(ABC):
    """Абстракция для сохранения результатов обработки"""

    @abstractmethod
    def save(self, image: Any, path: str) -> None:
        """Сохраняет результат обработки в файл"""
        pass