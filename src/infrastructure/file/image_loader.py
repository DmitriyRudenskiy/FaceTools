"""
Модуль для загрузки изображений из директории.

Содержит класс ImageLoader для поиска и валидации графических файлов.
"""

import os
from typing import List, Tuple, Optional
from PIL import Image, UnidentifiedImageError  # type: ignore

# Константа с поддерживаемыми расширениями
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


class ImageLoader:
    """Класс для загрузки графических файлов из директории."""

    def __init__(self, valid_extensions: Optional[Tuple[str, ...]] = None):
        """
        Инициализирует загрузчик изображений.

        Args:
            valid_extensions: Кортеж допустимых расширений файлов
        """
        self.valid_extensions: Tuple[str, ...] = (
            valid_extensions or SUPPORTED_EXTENSIONS
        )
        # Приводим расширения к нижнему регистру для единообразия
        self.valid_extensions = tuple(ext.lower() for ext in self.valid_extensions)

    def load_images(self, directory: str) -> List[str]:
        """
        Загружает графические файлы из указанной директории.

        Args:
            directory: Путь к директории с файлами

        Returns:
            Список полных путей к валидным изображениям
        """
        # Получаем абсолютный путь к директории
        abs_directory = os.path.abspath(directory)

        # Проверяем, существует ли директория
        if not os.path.exists(abs_directory):
            raise FileNotFoundError(f"Директория не найдена: {abs_directory}")

        if not os.path.isdir(abs_directory):
            raise NotADirectoryError(
                f"Указанный путь не является директорией: {abs_directory}"
            )

        image_paths: List[str] = []

        for entry in os.listdir(abs_directory):
            # Формируем полный путь к файлу
            full_path = os.path.join(abs_directory, entry)

            # Пропускаем поддиректории
            if os.path.isdir(full_path):
                continue

            # Проверка расширения файла
            ext = os.path.splitext(entry)[1].lower()
            if ext not in self.valid_extensions:
                continue

            # Проверка целостности изображения
            if not self._is_valid_image(full_path):
                continue

            image_paths.append(full_path)

        return image_paths

    def _is_valid_image(self, path: str) -> bool:
        """
        Проверяет, является ли файл валидным изображением.

        Args:
            path: Полный путь к файлу

        Returns:
            True если файл является валидным изображением, иначе False
        """
        try:
            with Image.open(path) as img:
                img.verify()  # Проверяем целостность без полной загрузки
            return True
        except (UnidentifiedImageError, IOError, OSError):
            return False
