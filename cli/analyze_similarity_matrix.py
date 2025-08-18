import os
import sys
from pathlib import Path

# Улучшенная обработка добавления корневой директории в sys.path
if __name__ == "__main__":
    # Определяем текущий файл и пытаемся найти корневую директорию проекта
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    # Ищем корневую директорию, поднимаясь вверх до тех пор, пока не найдем признак проекта
    root_dir = current_dir
    while True:
        # Проверяем наличие признаков корневой директории
        if (root_dir / "src").exists() and (root_dir / "cli").exists():
            break
        # Поднимаемся на уровень выше
        parent_dir = root_dir.parent
        if parent_dir == root_dir:
            raise RuntimeError("Не удалось найти корневую директорию проекта")
        root_dir = parent_dir
    # Добавляем корневую директорию в sys.path
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

from src.infrastructure.clustering.matrix_based_clusterer import CompareMatrix

if __name__ == "__main__":
    matrix = CompareMatrix(0)
    matrix.load('./debug.json')
    matrix.display_full()
    # Разделение по порогу
    threshold = 0.7
    groups = matrix.split(threshold)
    print(f"\nГруппы при пороге {threshold}:")
    for i, group in enumerate(groups):
        print(f"Группа {i + 1}: {group}")

        # Показываем имена файлов в группе
        group_files = ['' for idx in group]
        group_names = [os.path.basename(path) for path in group_files]
        print(f"  Файлы: {group_names}")

    # Получение подматриц с легендами
    submatrices = matrix.split_to_matrices(threshold)
    print("\nПодматрицы с легендами:")
    for i, submatrix_info in enumerate(submatrices):
        print(f"Подматрица {i+1}:")
        print(f"  Индексы: {submatrix_info['indices']}")
        #print(f"  Файлы: {[os.path.basename(path) for path in submatrix_info['legend']]}")
        print("  Матрица:")
        print(submatrix_info['matrix'])
        print()