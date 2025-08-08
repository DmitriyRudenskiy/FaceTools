import sys
import os
import glob
import argparse

# Добавляем путь к src директории
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.expert_deepface import ExpertDeepFaceAndArcFace
from utils.compare_matrix import CompareMatrix


def get_image_files_from_directory(directory_path):
    """
    Получает список графических файлов из директории

    Args:
        directory_path (str): Путь к директории с изображениями

    Returns:
        list: Список путей к графическим файлам
    """
    # Поддерживаемые расширения графических файлов
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']

    image_paths = []

    # Поиск файлов в указанной директории
    for extension in image_extensions:
        # Поиск файлов в основной директории
        files = glob.glob(os.path.join(directory_path, extension))
        image_paths.extend(files)

        # Поиск файлов во всех поддиректориях
        files = glob.glob(os.path.join(directory_path, '**', extension), recursive=True)
        image_paths.extend(files)

    # Удаление дубликатов и сортировка
    image_paths = list(set(image_paths))
    image_paths.sort()

    return image_paths


def main():
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description='Face recognition and clustering')
    parser.add_argument('-s', '--src', type=str, help='Path to directory with images')
    parser.add_argument('-t', '--threshold', type=float, default=0.7, help='Threshold for clustering (default: 0.7)')

    args = parser.parse_args()

    # Проверяем, что аргумент src предоставлен
    if not args.src:
        print("Ошибка: Необходимо указать путь к директории с изображениями")
        print("Использование: python script.py --src /path/to/images")
        return

    # Создаем экземпляр компаратора
    comparator = ExpertDeepFaceAndArcFace(
        detector_backend="retinaface",
        distance_metric="cosine"
    )

    # Получаем путь к директории из аргументов
    images_directory = args.src

    # Проверяем существование директории
    if not os.path.exists(images_directory):
        print(f"Директория не найдена: {images_directory}")
        return

    # Получаем список всех графических файлов из директории
    image_paths = get_image_files_from_directory(images_directory)

    # Склеиваем полный путь к каждому файлу
    image_paths = [os.path.join(os.path.abspath(images_directory), os.path.basename(path)) if not os.path.isabs(path) else os.path.abspath(path) for path in image_paths]

    # Проверяем, что файлы найдены
    if not image_paths:
        print(f"Не найдено графических файлов в директории: {images_directory}")
        return

    for i, path in enumerate(image_paths):
        print(f"{i + 1}. {os.path.basename(path)}")

    # Инициализируем с изображениями
    comparator.init(image_paths)
    print(f"\nКоличество успешно обработанных изображений: {len(comparator.storage)}")

    # Сравниваем первые два изображения (если есть хотя бы 2)
    if len(comparator.storage) >= 2:
        distance = comparator.compare(0, 1)
        print(f"\nРасстояние между первыми двумя изображениями: {distance}")

        # Верификация
        result = comparator.verify(distance)
        print(f"Верификация: {result}")

    # Создаем и заполняем матрицу сравнений
    if len(comparator.storage) > 0:
        matrix = CompareMatrix(len(comparator.storage))
        matrix.fill(comparator)
        matrix.save('./debug.json')
        matrix.display()

if __name__ == "__main__":
    main()