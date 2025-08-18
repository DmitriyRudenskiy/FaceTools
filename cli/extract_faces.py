#!/usr/bin/env python3
"""Точка входа CLI-приложения для кластеризации лиц.
Использование:
python -m application.cli.run <input_path> [output_dir]
python /path/to/extract_faces.py <input_path> [output_dir]
"""
import sys


def main():
    """Основная функция запуска приложения"""
    from src.application.cli.argument_parser import setup_project_environment

    # Настройка окружения
    setup_project_environment()

    try:
        # Теперь импортируем после настройки пути
        from src.config.dependency_injector import DependencyInjector
    except ImportError as e:
        print(f"[ERROR] Не удалось импортировать зависимости: {str(e)}")
        print("Возможные причины:")
        print("1. Отсутствуют необходимые файлы в проекте")
        print("2. Неправильная структура проекта")
        print("3. Отсутствуют зависимости (установите через pip install -r requirements.txt)")
        return 2

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(
            "Использование: python -m application.cli.run <путь_к_изображению/директории> [директория_для_сохранения]")
        print("Примеры:")
        print(" python -m application.cli.run ./photos")
        print(" python -m application.cli.run ./photos ./results")
        return 1

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) == 3 else None

    try:
        injector = DependencyInjector()
        service = injector.get_face_clustering_service()
        return 0 if service.process_images(input_path, output_dir) else 1
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        import traceback
        print(f"Детали ошибки:{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())