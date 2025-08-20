"""
Command-line script for creating image collages from a directory of images.
"""

import argparse
import os
import sys
from typing import List
from PIL import Image  # type: ignore

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.infrastructure.file.image_loader import ImageLoader
from src.utils.collage_creator import CollageCreator

def main() -> None:
    """Основная функция для запуска скрипта из командной строки."""
    parser = argparse.ArgumentParser(description='Создание коллажей из изображений')
    parser.add_argument('-s', '--source', required=True, help='Путь к директории с изображениями')
    parser.add_argument('-o', '--output', default='.', help='Директория для сохранения результатов')
    args = parser.parse_args()

    try:
        # Загружаем изображения
        loader = ImageLoader()
        images = loader.load_images(args.source)

        if not images:
            print("В указанной директории не найдено изображений")
            return

        print(f"Загружено {len(images)} изображений")

        # Process images - ensure they are PIL Image objects
        processed_images: List[Image.Image] = []
        for img in images:
            if isinstance(img, str):
                # Open image if path is provided
                processed_images.append(Image.open(img))
            elif hasattr(img, 'width'):  # Basic PIL Image check
                processed_images.append(img)
            else:
                print(f"Пропущен неподдерживаемый тип: {type(img)}")
                continue

        # Создаем коллажи
        creator = CollageCreator()
        creator.create_collages(processed_images, args.output)

    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Ошибка: {e}")
    except OSError as e:
        # More specific exception handling for file/system errors
        print(f"Системная ошибка: {e}")

if __name__ == "__main__":
    main()
