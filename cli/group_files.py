import json
import argparse
import os
import sys

# Add project root to path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.infrastructure.persistence.group_organizer import GroupOrganizer
from collections import defaultdict


class ImageGrouper:
    """Класс для группировки изображений на основе матрицы расстояний."""

    def __init__(self, legend, matrix, threshold=0.5):
        """
        Инициализирует ImageGrouper.

        Args:
            legend (list): Список путей к файлам.
            matrix (list): Матрица расстояний.
            threshold (float): Пороговое значение для определения принадлежности к группе.
        """
        self.legend = legend
        self.matrix = matrix
        self.threshold = threshold
        self.groups = []

    def group_images(self):
        """
        Группирует изображения на основе матрицы расстояний.

        Returns:
            list: Список групп с данными о изображениях.
        """
        # Словарь для отслеживания, к какой группе принадлежит каждое изображение
        image_to_group = {}
        # Счетчик для ID групп
        group_id = 0

        # Проходим по всем изображениям
        for i in range(len(self.legend)):
            if i not in image_to_group:
                # Если изображение еще не в группе, создаем новую группу
                current_group = {
                    'id': group_id,
                    'representative': os.path.basename(self.legend[i]),
                    'images_full_paths': [self.legend[i]],
                    'indices': [i]
                }

                # Проверяем, какие другие изображения могут принадлежать этой группе
                for j in range(i + 1, len(self.legend)):
                    if j not in image_to_group and self.matrix[i][j] <= self.threshold:
                        current_group['images_full_paths'].append(self.legend[j])
                        current_group['indices'].append(j)
                        image_to_group[j] = group_id

                # Помечаем представителя группы
                image_to_group[i] = group_id
                self.groups.append(current_group)
                group_id += 1

        return self.groups


def load_json_data(json_file_path):
    """
    Загружает данные из JSON файла.

    Args:
        json_file_path (str): Путь к JSON файлу.

    Returns:
        tuple: Кортеж из legend и matrix.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data['legend'], data['matrix']


def main():
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(description='Организация файлов по группам на основе матрицы расстояний.')
    parser.add_argument('json_file', help='Путь к JSON файлу с матрицей расстояний')
    parser.add_argument('output_dir', help='Каталог для сохранения сгруппированных файлов')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Пороговое значение для группировки (по умолчанию: 0.5)')

    args = parser.parse_args()

    # Проверяем существование JSON файла
    if not os.path.exists(args.json_file):
        print(f"Ошибка: Файл {args.json_file} не найден.")
        return 1

    try:
        # Загружаем данные из JSON
        print("Загрузка данных из JSON файла...")
        legend, matrix = load_json_data(args.json_file)
        print(f"Загружено {len(legend)} изображений.")

        # Создаем объект для группировки
        print("Группировка изображений...")
        grouper = ImageGrouper(legend, matrix, args.threshold)
        groups_data = grouper.group_images()
        print(f"Сформировано {len(groups_data)} групп.")

        # Создаем объект для организации файлов
        organizer = GroupOrganizer(groups_data, args.output_dir)

        # Организуем файлы по группам
        organizer.organize()

        print(f"Файлы успешно организованы в каталоге: {args.output_dir}")
        return 0

    except Exception as e:
        print(f"Ошибка при выполнении скрипта: {e}")
        return 1


if __name__ == "__main__":
    exit(main())