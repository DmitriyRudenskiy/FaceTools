import time
import argparse
from application.use_cases import ClusterFacesUseCase


def main():
    """Точка входа в программу."""
    parser = argparse.ArgumentParser(description='Анализ и группировка изображений по схожести лиц')
    parser.add_argument('-s', '--src', required=True, help='Путь к директории с изображениями для анализа')
    parser.add_argument('-o', '--output', default='groups.json', help='Путь к выходному JSON файлу')
    parser.add_argument('-d', '--dest', help='Путь к директории для создания подкаталогов с группами')
    parser.add_argument('-m', '--show-matrix', action='store_true', help='Отображать матрицу схожести в консоли')
    parser.add_argument('-r', '--show-reference-table', action='store_true',
                        help='Отображать таблицу сопоставления с эталонами')
    parser.add_argument('--max-clusters', type=int, default=20, help='Максимальное количество кластеров')
    parser.add_argument('--method', choices=['silhouette', 'elbow'], default='silhouette',
                        help='Метод определения оптимального количества кластеров')
    parser.add_argument('--ctx-id', type=int, default=0, help='ID устройства (-1 для CPU, 0 для GPU)')
    parser.add_argument('--det-size', type=str, default='640,640', help='Размер детекции (ширина,высота)')
    parser.add_argument('--det-thresh', type=float, default=0.5, help='Порог детекции')
    parser.add_argument('--no-json', action='store_true', help='Не сохранять JSON файл')
    parser.add_argument('--no-images', action='store_true', help='Не сохранять изображения в директории')

    args = parser.parse_args()

    # Обработка параметров
    det_size = tuple(map(int, args.det_size.split(',')))

    # Создаем use case через инъекцию зависимостей
    use_case = ClusterFacesUseCase.create_default(
        ctx_id=args.ctx_id,
        det_size=det_size,
        det_thresh=args.det_thresh
    )

    # Запускаем обработку
    total_start_time = time.time()

    try:
        result = use_case.execute(
            input_dir=args.src,
            output_json=not args.no_json,
            output_json_path=args.output,
            organize_files=not args.no_images and args.dest is not None,
            dest_dir=args.dest,
            max_clusters=args.max_clusters,
            method=args.method
        )

        # Выводим результаты
        elapsed_time = time.time() - total_start_time
        print(f"\n=== Обработка завершена ===")
        print(f"Найдено лиц: {result['total_faces']}")
        print(f"Сформировано групп: {result['total_groups']}")
        print(f"Время выполнения: {elapsed_time:.2f} секунд")

        if args.show_matrix:
            use_case.display_similarity_matrix()

        if args.show_reference_table:
            use_case.display_reference_table()

    except Exception as e:
        print(f"Ошибка при выполнении: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()