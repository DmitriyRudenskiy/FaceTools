"""
Module for creating image collages from a list of images.
"""

import os
import random
from typing import List
from PIL import Image  # type: ignore


class CollageCreator:
    """Класс для создания коллажей из изображений."""

    # Константы для размеров сетки коллажей
    GRID_SIZE_2x2 = 2
    GRID_SIZE_3x3 = 3
    GRID_SIZE_4x4 = 4
    GRID_SIZES = [GRID_SIZE_2x2, GRID_SIZE_3x3, GRID_SIZE_4x4]

    def __init__(self, target_size: int = 1024) -> None:
        """
        Инициализирует создатель коллажей.

        Args:
            target_size: Размер конечного изображения в пикселях
        """
        self.target_size = target_size
        self.errors: List[str] = []  # Список для хранения ошибок

    def create_collage(self, images: List[Image.Image], grid_size: int) -> Image.Image:
        """
        Создает коллаж из изображений с указанной сеткой.

        Args:
            images: Список изображений для коллажа
            grid_size: Размер сетки (2, 3 или 4)

        Returns:
            Объект PIL.Image с коллажем
        """
        cell_size = self.target_size // grid_size
        extra_pixels = self.target_size % grid_size

        collage = Image.new('RGB', (self.target_size, self.target_size))

        for i in range(grid_size):
            for j in range(grid_size):
                # Вычисляем размеры ячейки с учетом остаточных пикселей
                width = cell_size + (1 if j == grid_size - 1 and extra_pixels else 0)
                height = cell_size + (1 if i == grid_size - 1 and extra_pixels else 0)

                img_index = i * grid_size + j
                if img_index < len(images):
                    img = images[img_index]
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                    collage.paste(img, (j * cell_size, i * cell_size))

        return collage

    def create_collages(self, images: List[Image.Image], output_dir: str = ".") -> List[str]:
        """
        Создает коллажи различных размеров из предоставленных изображений.

        Args:
            images: Список изображений для создания коллажей
            output_dir: Директория для сохранения результатов

        Returns:
            Список путей к созданным файлам коллажей
        """
        # Очищаем предыдущие ошибки перед новым процессом создания
        self.errors = []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        created_paths: List[str] = []  # Список для хранения путей к созданным коллажам

        for grid_size in self.GRID_SIZES:
            n = grid_size * grid_size
            if len(images) < n:
                # Сохраняем ошибку вместо прямого вывода
                error_msg = (
                    f"Пропущена сетка {grid_size}x{grid_size} "
                    f"(недостаточно изображений)"
                )
                self.errors.append(error_msg)
                continue

            selected = random.sample(images, n)
            collage = self.create_collage(selected, grid_size)
            output_path = os.path.join(output_dir, f"collage_{grid_size}x{grid_size}.jpg")
            collage.save(output_path)
            created_paths.append(output_path)  # Добавляем путь в список

        return created_paths

    def show_errors(self) -> None:
        """Выводит все сохранённые ошибки на экран."""
        for error in self.errors:
            print(error)
