import sys
import argparse
import os
import json
import time
import shutil

# Add project root to path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# pylint: disable=import-error,wrong-import-position
from src.domain.сompare_matrix import CompareMatrix
from src.infrastructure.clustering.legacy_image_grouper import ImageGrouper
# pylint: enable=import-error,wrong-import-position


def _organize_files(clusters, destination_directory):
    """Организует файлы по группам в отдельные каталоги."""
    start_time = time.time()
    total_copied = 0

    for cluster in clusters:
        # Используем имя файла представителя (без расширения) как имя каталога
        representative_name = os.path.splitext(cluster.representative)[0]
        # Очищаем имя от недопустимых символов для имен файлов/каталогов
        safe_group_name = "".join(c for c in representative_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        # Если имя оказалось пустым, используем ID группы
        if not safe_group_name:
            safe_group_name = f"Group_{cluster.id}"
        group_directory_path = os.path.join(destination_directory, safe_group_name)
        print(f"Создаю каталог для группы {cluster.id}: {group_directory_path}")
        # Создаем каталог для группы
        try:
            os.makedirs(group_directory_path, exist_ok=True)
        except OSError as e:
            print(f"Ошибка создания каталога {group_directory_path}: {e}")
            continue  # Пропускаем эту группу, если не удалось создать каталог

        # Копируем файлы группы в созданный каталог
        copied_count = 0
        for full_path in cluster.members_paths:
            try:
                filename = os.path.basename(full_path)
                destination_file_path = os.path.join(group_directory_path, filename)
                # Копируем файл
                shutil.copy2(full_path, destination_file_path)
                copied_count += 1
                total_copied += 1
            except Exception as e:
                print(f"Ошибка копирования файла {full_path} в {group_directory_path}: {e}")
        print(f"  Скопировано файлов в группу '{safe_group_name}': {copied_count}")

    print(f"=== Организация файлов завершена ===")


def _save_results(clustering_result, output_file):
    """Сохраняет результаты в JSON файл."""

    # Формируем итоговый JSON
    result_json = {
        "timestamp": clustering_result.timestamp,
        "total_groups": clustering_result.total_clusters,
        "unrecognized_count": clustering_result.unrecognized_count,
        "groups": [
            {
                "id": cluster.id,
                "size": cluster.size,
                "representative": cluster.representative,
                "representative_full_path": cluster.representative_path,
                "images": cluster.members,
                "images_full_paths": cluster.members_paths,
                "average_similarity": cluster.average_similarity
            }
            for cluster in clustering_result.clusters
        ],
        "unrecognized_images": clustering_result.unrecognized_images
    }

    # Сохраняем в файл
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Ошибка при сохранении файла {output_file}: {e}")

    print(f"=== Анализ завершен ===")
    print(f"Результаты сохранены в файл: {output_file}")
    print(f"Найдено групп: {clustering_result.total_clusters}")
    print(f"Нераспознанных изображений: {clustering_result.unrecognized_count}")


def main():
    """Основная точка входа для CLI."""
    parser = argparse.ArgumentParser(
        description="Кластеризация лиц по схожести с использованием DeepFace.",
        formatter_class=argparse.RawTextHelpFormatter  # Для корректного отображения \n в help
    )

    parser.add_argument('-j', '--json', default='matrix.json', help='Путь к JSON файлу с матрицей сравнения')
    parser.add_argument('-o', '--out', default='groups.json', help='Путь к JSON файлу с матрицей сравнения')
    parser.add_argument('-d', '--dest', default='groups', help='Путь к JSON файлу с матрицей сравнения')

    args = parser.parse_args()

    # Проверяем существование файла
    if not os.path.exists(args.json):
        print(f"Файл {args.json} не найден")
        return 1

    matrix_instance = CompareMatrix.from_json(args.json)

    print(f"Загружено {len(matrix_instance.legend)}x{len(matrix_instance.legend)}")

    """Группирует изображения по схожести лиц."""
    grouper = ImageGrouper(matrix_instance.matrix, matrix_instance.legend)
    clustering_result = grouper.cluster()

    # Выводим информацию о группах
    print("\nГруппировка завершена:")

    # Сохраняем результаты в JSON
    _save_results(clustering_result, args.out)

    # Организуем файлы по группам, если указан путь назначения
    if args.dest and clustering_result.clusters:
        _organize_files(clustering_result.clusters, args.dest)

    return 0


if __name__ == "__main__":
    sys.exit(main())