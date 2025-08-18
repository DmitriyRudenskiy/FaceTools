#!/usr/bin/env python3
"""Инструмент для кластеризации лиц по схожести с использованием DeepFace.
Использование: python cli/deepface_face_cluster.py -s <source_dir> [options]"""

# --- Стандартные импорты ---
import sys
import argparse
import os
import glob
import time
import json
from pathlib import Path
from typing import List, Tuple
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

# Добавляем корневую директорию в sys.path, если она еще не добавлена
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
    # print(f"[DEBUG] Добавлена корневая директория в sys.path: {root_dir}")

# Проверяем, что нужные модули доступны
try:
    # --- Импорты из проекта ---
    # Импортируем правильный DeepFaceFaceComparator
    from src.infrastructure.comparison.deepface_comparator import DeepFaceFaceComparator
    from src.infrastructure.clustering.matrix_based_clusterer import CompareMatrix
    from src.domain.cluster import ClusteringResult
    from src.infrastructure.persistence.group_organizer import GroupOrganizer
except ImportError as e:
    print(f"[ERROR] Не удалось импортировать зависимости: {str(e)}")
    print("Возможные причины:")
    print("1. Неправильно определена корневая директория проекта")
    print("2. Отсутствует файл __init__.py в корневой директории")
    print("3. Структура проекта не соответствует ожидаемой")
    sys.exit(1)


# --- Функции и основная логика ---

def get_image_files_from_directory(directory_path: str, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')) -> List[str]:
    """Получает список всех графических файлов в директории и поддиректориях."""
    image_paths = []
    for extension in extensions:
        # Используем glob для рекурсивного поиска
        files = glob.glob(os.path.join(directory_path, '**', f'*{extension}'), recursive=True)
        image_paths.extend(files)
    # Удаление дубликатов и сортировка
    image_paths = list(set(image_paths))
    image_paths.sort()
    return image_paths


def _compare_by_index(comparator, index1, index2):
    """
    Сравнивает два изображения по индексам, используя косинусное расстояние.
    Возвращает кортеж (совпадение: bool, расстояние: float).
    """
    try:
        # Получаем эмбеддинги по индексам
        emb1 = comparator.storage[index1]
        emb2 = comparator.storage[index2]

        # Вычисляем косинусное расстояние
        # np.dot для нормализованных векторов дает косинус угла
        dot_product = np.dot(emb1, emb2)
        # Преобразуем косинус в "расстояние": 0 (полное совпадение) -> 1 (полная противоположность)
        face_distance = (1 - dot_product) / 2

        # Порог для определения совпадения (например, 0.3 для ArcFace)
        threshold = 0.35
        is_match = face_distance < threshold

        return [is_match, face_distance]

    except Exception as e:
        print(f"Ошибка при сравнении лиц {index1} и {index2}: {e}")
        # Возвращаем False и максимальное расстояние в случае ошибки
        return [False, 1.0]


def main():
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(description="Кластеризация лиц с использованием DeepFace.")
    parser.add_argument('-s', '--src', required=True, help='Путь к директории с изображениями')
    parser.add_argument('-o', '--output', default='deepface_groups.json',
                        help='Путь к выходному JSON файлу (по умолчанию: deepface_groups.json)')
    parser.add_argument('-d', '--dest', help='Директория для организации файлов по группам')
    parser.add_argument('-t', '--threshold', type=float, default=0.7,
                        help='Порог для кластеризации (по умолчанию: 0.7)')
    parser.add_argument('-m', '--show-matrix', action='store_true', help='Отображать матрицу схожести в консоли')
    parser.add_argument('-r', '--references', action='store_true',
                        help="Отображать таблицу сопоставления с эталонами (файлы, начинающиеся с 'refer_')")
    args = parser.parse_args()

    # Проверяем существование директории
    if not os.path.exists(args.src):
        print(f"Директория не найдена: {args.src}")
        return 1

    # Получаем список всех графических файлов из директории
    image_paths = get_image_files_from_directory(args.src)

    # Проверяем, что файлы найдены
    if not image_paths:
        print(f"Не найдено графических файлов в директории: {args.src}")
        return 1

    print(f"Количество найденных изображений: {len(image_paths)}")
    # print("Найденные изображения:")
    # for i, path in enumerate(image_paths):
    #     print(f"{i + 1}. {os.path.basename(path)}")

    # Создаем и заполняем матрицу сравнений
    try:
        print("Инициализация компаратора DeepFace...")
        # Создаем компаратор
        # Убедитесь, что модель ArcFace доступна для DeepFace
        comparator = DeepFaceFaceComparator(detector_backend="retinaface", distance_metric="cosine")

        print("Загрузка изображений и вычисление эмбеддингов...")
        start_time = time.time()
        # Инициализируем хранилище эмбеддингов
        comparator.init(image_paths)
        loading_time = time.time() - start_time
        print(f"Загрузка завершена за {loading_time:.2f} секунд.")

        if not comparator.storage or len(comparator.storage) != len(image_paths):
             print("Ошибка: Не удалось загрузить все эмбеддинги.")
             return 1

        print("Создание матрицы схожести...")
        matrix_start_time = time.time()
        # Создаем объект CompareMatrix
        matrix = CompareMatrix(len(image_paths))
        matrix.image_paths = image_paths # Устанавливаем пути к изображениям

        # Заполняем матрицу схожести, используя нашу функцию сравнения
        num_images = len(image_paths)
        for i in range(num_images):
            for j in range(i, num_images): # Заполняем верхнюю треугольную часть (включая диагональ)
                if i == j:
                    # Расстояние от изображения до самого себя
                    matrix.set_value(i, j, [True, 0.0])
                else:
                    # Сравниваем изображения
                    result = _compare_by_index(comparator, i, j)
                    matrix.set_value(i, j, result)
                    matrix.set_value(j, i, result) # Матрица симметрична

        matrix_elapsed_time = time.time() - matrix_start_time
        print(f"Матрица схожести создана за {matrix_elapsed_time:.2f} секунд.")

        matrix.legend = [image_path for embedding, image_path in comparator.storage]

        # Отображаем матрицу, если запрошено
        if args.show_matrix:
             print("\n--- Матрица схожести ---")
             matrix._print_similarity_matrix()

        # Отображаем таблицу с эталонами, если запрошено
        if args.references:
             print("\n--- Таблица сопоставления с эталонами ---")
             matrix._print_reference_table()

        # Кластеризация
        print("\nВыполняется кластеризация...")
        cluster_start_time = time.time()
        # Группируем изображения
        groups_data = matrix.group_images(args.threshold)
        cluster_elapsed_time = time.time() - cluster_start_time
        print(f"Кластеризация завершена за {cluster_elapsed_time:.2f} секунд.")

        # Выводим информацию о группах
        print("\n--- Результаты группировки ---")
        if groups_data:
            for group_data in groups_data:
                print(f"Группа {group_data['id']} (представлена {group_data['representative']}):")
                for path in group_data['images']:
                    print(f"  {path}")
                print()
            print(f"Найдено групп: {len(groups_data)}")
        else:
            print("Группы не найдены.")

        # Подсчет нераспознанных изображений
        recognized_images_set = set()
        for group_data in groups_data:
            recognized_images_set.update(group_data['images'])
        all_images_set = set(os.path.basename(p) for p in image_paths)
        unrecognized_images_set = all_images_set - recognized_images_set
        unrecognized_count = len(unrecognized_images_set)

        # Создаем объект ClusteringResult для передачи в GroupOrganizer
        # (Здесь упрощенная реализация, в реальности может потребоваться больше данных)
        class MockClusteringResult:
             def __init__(self, groups_data, unrecognized_count, unrecognized_images_set):
                 self.clusters = groups_data # Для совместимости с GroupOrganizer
                 self.total_clusters = len(groups_data)
                 self.unrecognized_count = unrecognized_count
                 self.unrecognized_images = list(unrecognized_images_set) # Простой список имен файлов
                 self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        clustering_result = MockClusteringResult(groups_data, unrecognized_count, unrecognized_images_set)

        # Сохраняем результаты в JSON
        print(f"\nСохранение результатов в {args.output}...")
        try:
            # Формируем итоговый JSON
            result_json = {
                "timestamp": clustering_result.timestamp,
                "total_groups": clustering_result.total_clusters,
                "unrecognized_count": clustering_result.unrecognized_count,
                "groups": groups_data, # groups_data уже содержит необходимую структуру
                "unrecognized_images": clustering_result.unrecognized_images
            }

            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, ensure_ascii=False, indent=4)
            print(f"Результаты успешно сохранены в файл: {args.output}")

        except Exception as e:
            print(f"Ошибка при сохранении файла {args.output}: {e}")
            return 1

        # Организуем файлы по группам, если указан путь назначения
        if args.dest:
            if clustering_result.clusters:
                print(f"\nОрганизация файлов по группам в директорию: {args.dest}...")
                try:
                    organizer = GroupOrganizer(clustering_result.clusters, args.dest)
                    organizer.organize()
                    print("Файлы успешно организованы по группам.")
                except Exception as e:
                    print(f"Ошибка при организации файлов: {e}")
                    # Не завершаем работу из-за ошибки организации файлов
            else:
                print("\nОрганизация файлов не будет выполнена, так как не найдено групп.")

        print("\n--- Анализ завершен ---")
        total_elapsed_time = time.time() - start_time
        print(f"Общее время выполнения: {total_elapsed_time:.2f} секунд")
        print(f"Обработано изображений: {len(image_paths)}")
        print(f"Найдено групп: {clustering_result.total_clusters}")
        print(f"Нераспознанных изображений: {clustering_result.unrecognized_count}")

        return 0 # Успешное завершение

    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        import traceback
        print(f"Детали ошибки:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    # sys.path уже настроен в начале файла
    sys.exit(main())