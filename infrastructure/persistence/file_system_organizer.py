from core.interfaces.file_organizer import FileOrganizer
from core.interfaces.result_saver import ResultSaver
import os
from typing import List  # Добавлен импорт List


class FileSystemOrganizer(FileOrganizer, ResultSaver):
    """Работа с файловой системой и сохранение результатов"""

    def get_image_files(self, path: str) -> List[str]:
        """Возвращает список путей к изображениям в указанном пути"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}

        if not self.exists(path):
            return []

        if not self.is_directory(path):
            ext = os.path.splitext(path)[1].lower()
            return [path] if ext in image_extensions else []

        return [
            os.path.join(root, file)
            for root, _, files in os.walk(path)
            for file in files
            if os.path.splitext(file)[1].lower() in image_extensions
        ]

    def create_directory(self, path: str) -> None:
        """Создает директорию, если она не существует"""
        os.makedirs(path, exist_ok=True)

    def exists(self, path: str) -> bool:
        """Проверяет существование пути"""
        return os.path.exists(path)

    def is_directory(self, path: str) -> bool:
        """Проверяет, является ли путь директорией"""
        return os.path.isdir(path)

    def get_directory(self, path: str) -> str:
        """Возвращает директорию, содержащую файл"""
        return os.path.dirname(path)

    def get_basename(self, path: str) -> str:
        """Возвращает базовое имя файла без расширения"""
        return os.path.splitext(os.path.basename(path))[0]

    def save(self, image: any, path: str) -> None:
        """Сохраняет изображение в файл"""
        dir_path = os.path.dirname(path)
        if dir_path and not self.exists(dir_path):
            self.create_directory(dir_path)

        image.save(path)