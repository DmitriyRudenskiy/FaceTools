import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import List

# Импорт numpy для использования np.dot
import numpy as np

# --- Добавление корневой директории в sys.path ---
# Это необходимо сделать ДО любых импортов из src.*
# Определяем текущий файл и пытаемся найти корневую директорию проекта
current_file = Path(__file__).resolve()
current_dir = current_file.parent
# Ищем корневую директорию, поднимаясь вверх до тех пор, пока не найдем признак проекта
root_dir = current_dir
max_iterations = 10 # Защита от бесконечного цикла
iterations = 0
while iterations < max_iterations:
    # Проверяем наличие признаков корневой директории проекта
    src_dir = root_dir / "src"
    has_core = (src_dir / "core").is_dir()
    has_domain = (src_dir / "domain").is_dir()
    has_infrastructure = (src_dir / "infrastructure").is_dir()
    if has_core and has_domain and has_infrastructure:
        break
    # Поднимаемся на уровень выше
    parent_dir = root_dir.parent
    if parent_dir == root_dir: # Достигли корня файловой системы
        print("[ERROR] Не удалось найти корневую директорию проекта. Убедитесь, что структура проекта корректна.")
        sys.exit(1)
    root_dir = parent_dir
    iterations += 1

# Add project root to path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.infrastructure.file.image_loader import ImageLoader
from src.domain.сompare_matrix import CompareMatrix
from src.infrastructure.comparison.deepface_comparator import DeepFaceComparator

# Добавляем корневую директорию в sys.path, если она еще не добавлена
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
    print(f"[DEBUG] Добавлена корневая директория в sys.path: {root_dir}")

def main():
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(description="Кластеризация лиц с использованием DeepFace.")
    parser.add_argument('-s', '--src', required=True, help='Путь к директории с изображениями')
    parser.add_argument('-o', '--output', default='matrix.json',
                        help='Путь к выходному JSON файлу (по умолчанию: matrix.json)')
    parser.add_argument('-d', '--dest', help='Директория для организации файлов по группам')
    parser.add_argument('-t', '--threshold', type=float, default=0.7,
                        help='Порог для кластеризации (по умолчанию: 0.7)')
    parser.add_argument('-m', '--show-matrix', action='store_true', help='Отображать матрицу схожести в консоли')
    parser.add_argument('-r', '--references', action='store_true',
                        help="Отображать таблицу сопоставления с эталонами (файлы, начинающиеся с 'refer_')")
    args = parser.parse_args()

    loader = ImageLoader()

    # Получаем список всех графических файлов из директории
    image_paths = loader.load_images(args.src)

    # Проверяем, что файлы найдены
    if not image_paths:
        print(f"Не найдено графических файлов в директории: {args.src}")
        return 1

    print(f"Количество найденных изображений: {len(image_paths)}")
    print("Найденные изображения:")
    for i, path in enumerate(image_paths):
        print(f"{i + 1}. {os.path.basename(path)}")

    print("Инициализация компаратора DeepFace...")

    comparator = DeepFaceComparator()
    load_image_paths = comparator.init(image_paths)
    num_images = len(load_image_paths)

    if not num_images:
        print(f"[WARNING] Загруженой 0 эмбеддингов")
        return None

    if num_images != len(image_paths):
        print(f"[WARNING] Не удалось загрузить все эмбеддинги. Загружено {num_images} из {len(image_paths)}")

    print("Создание матрицы схожести...")

    matrix = CompareMatrix(load_image_paths)

    for i in range(num_images):
        for j in range(i, num_images):
            if i == j:
                # Расстояние от изображения до самого себя
                matrix.set_value(i, j, 1.0)
            else:
                # Сравниваем изображения
                result = comparator.storage.compare_by_index(i, j)

                print(f"{i} | {j}  | {result:.4f}")

                matrix.set_value(i, j, result)

                # Матрица симметрична
                matrix.set_value(j, i, result)

    matrix.to_json(args.output)


if __name__ == "__main__":
    # sys.path уже настроен в начале файла
    sys.exit(main())