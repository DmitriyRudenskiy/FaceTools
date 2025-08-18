from abc import ABC, abstractmethod
from typing import List, Dict  # Добавлен импорт Dict


class FileOrganizer(ABC):
    """Абстракция для работы с файловой системой"""

    @abstractmethod
    def get_image_files(self, path: str) -> List[str]:
        """Возвращает список путей к изображениям в указанном пути"""
        pass

    @abstractmethod
    def create_directory(self, path: str) -> None:
        """Создает директорию, если она не существует"""
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Проверяет существование пути"""
        pass

    @abstractmethod
    def is_directory(self, path: str) -> bool:
        """Проверяет, является ли путь директорией"""
        pass

    @abstractmethod
    def get_directory(self, path: str) -> str:
        """Возвращает директорию, содержащую файл"""
        pass

    @abstractmethod
    def get_basename(self, path: str) -> str:
        """Возвращает базовое имя файла без расширения"""
        pass

    @abstractmethod
    def organize_by_clusters(self, clusters: List[Dict], destination: str) -> None:
        """Организует файлы по кластерам в отдельные директории"""
        pass
