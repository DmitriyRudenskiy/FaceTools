import sys
import argparse
import os

# Add project root to path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# pylint: disable=import-error,wrong-import-position
from src.domain.сompare_matrix import CompareMatrix
# pylint: enable=import-error,wrong-import-position


def main():
    """Основная точка входа для CLI."""
    parser = argparse.ArgumentParser(
        description="Кластеризация лиц по схожести с использованием DeepFace.",
        formatter_class=argparse.RawTextHelpFormatter  # Для корректного отображения \n в help
    )

    parser.add_argument('-j', '--json', default='groups.json', help='Путь к JSON файлу с матрицей сравнения')

    args = parser.parse_args()

    # Проверяем существование файла
    if not os.path.exists(args.json):
        print(f"Файл {args.json} не найден")
        return 1

    matrix_instance = CompareMatrix.from_json(args.json)

    print(f"Загружено {len(matrix_instance.legend)}x{len(matrix_instance.legend)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())