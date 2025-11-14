import argparse
import json
import os
import sys
import shutil
from pathlib import Path

# Импорт numpy для использования np.dot
import numpy as np

# --- Добавление корневой директории в sys.path ---
current_file = Path(__file__).resolve()
current_dir = current_file.parent
root_dir = current_dir
max_iterations = 10
iterations = 0
while iterations < max_iterations:
    src_dir = root_dir / "src"
    has_core = (src_dir / "core").is_dir()
    has_domain = (src_dir / "domain").is_dir()
    has_infrastructure = (src_dir / "infrastructure").is_dir()
    if has_core and has_domain and has_infrastructure:
        break
    parent_dir = root_dir.parent
    if parent_dir == root_dir:
        print("[ERROR] Не удалось найти корневую директорию проекта. Убедитесь, что структура проекта корректна.")
        sys.exit(1)
    root_dir = parent_dir
    iterations += 1

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.infrastructure.file.image_loader import ImageLoader
from src.domain.сompare_matrix import CompareMatrix
from src.infrastructure.comparison.deepface_comparator import DeepFaceComparator

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
    print(f"[DEBUG] Добавлена корневая директория в sys.path: {root_dir}")


def main():
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(description="Кластеризация лиц с использованием DeepFace.")
    parser.add_argument('-s', '--src', required=True, help='Путь к директории с изображениями')
    parser.add_argument('-j', '--json', default='matrix.json',
                        help='Путь к выходному JSON файлу (по умолчанию: matrix.json)')
    parser.add_argument('-o', '--out', default='matrix',
                        help='Директория для перемещения успешно распознанных изображений (по умолчанию: matrix)')
    parser.add_argument('-t', '--threshold', type=float, default=0.7,
                        help='Порог для кластеризации (по умолчанию: 0.7)')
    parser.add_argument('-m', '--show-matrix', action='store_true', help='Отображать матрицу схожести в консоли')
    parser.add_argument('-r', '--references', action='store_true',
                        help="Отображать таблицу сопоставления с эталонами (файлы, начинающиеся с 'refer_')")
    args = parser.parse_args()

    # Сканирование директории (только текущая директория)
    print(f"Сканирование директории {args.src} (без поддиректорий)...")

    # Поддерживаемые расширения изображений
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    # Получаем список файлов только из текущей директории
    image_paths = []
    try:
        for file in os.listdir(args.src):
            full_path = os.path.join(args.src, file)
            if os.path.isfile(full_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    image_paths.append(full_path)
    except FileNotFoundError:
        print(f"[ERROR] Указанная директория не существует: {args.src}")
        return 1
    except PermissionError:
        print(f"[ERROR] Нет прав для доступа к директории: {args.src}")
        return 1

    # Проверяем, что файлы найдены
    if not image_paths:
        print(f"Не найдено графических файлов в директории: {args.src}")
        print("Поддерживаемые форматы: .jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp")
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
        print(f"[WARNING] Загружено 0 эмбеддингов")
        return None

    if num_images != len(image_paths):
        print(f"[WARNING] Не удалось загрузить все эмбеддинги. Загружено {num_images} из {len(image_paths)}")

    print("Создание матрицы схожести...")

    matrix = CompareMatrix(load_image_paths)

    for i in range(num_images):
        for j in range(i, num_images):
            if i == j:
                matrix.set_value(i, j, 1.0)
            else:
                result = comparator.storage.compare_by_index(i, j)
                print(f"{i} | {j}  | {result:.4f}")
                matrix.set_value(i, j, result)
                matrix.set_value(j, i, result)

    # Сохраняем матрицу схожести
    matrix.to_json(args.json)

    # === Используем параметр out ===
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nПроверка и перемещение {num_images} успешно распознанных изображений в папку {out_dir}...")

    # Счетчики для статистики
    moved_count = 0
    skipped_count = 0

    for idx, src_path in enumerate(load_image_paths):
        src_path = Path(src_path).resolve()
        if not src_path.exists():
            print(f"  ✗ Файл {src_path.name} не существует, пропускаем")
            continue

        # Проверяем, находится ли файл уже в целевой директории
        if src_path.parent == out_dir:
            print(f"  → {src_path.name} уже находится в целевой директории, пропускаем")
            skipped_count += 1
            continue

        file_ext = src_path.suffix
        dest_path = out_dir / f"{idx}{file_ext}"

        try:
            shutil.move(str(src_path), str(dest_path))
            print(f"  ✓ {src_path.name} -> {dest_path.name} (index: {idx})")
            moved_count += 1
        except Exception as e:
            print(f"  ✗ Ошибка при перемещении {src_path.name}: {str(e)}")

    # Вывод итоговой статистики
    if skipped_count > 0:
        print(f"\nПропущено {skipped_count} файлов (уже находятся в целевой директории)")

    print(f"Успешно перемещено {moved_count} изображений в папку {out_dir}")
    print(f"Всего обработано: {num_images} изображений")


if __name__ == "__main__":
    sys.exit(main())