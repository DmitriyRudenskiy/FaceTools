import os

from utils.compare_matrix import CompareMatrix

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