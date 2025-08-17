#!/usr/bin/env python3
"""
Точка входа CLI-приложения для кластеризации лиц.
Использование:
  python -m application.cli.run <input_path> [output_dir]
  python /path/to/extract_faces.py <input_path> [output_dir]
"""
import sys
import os
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path при прямом запуске
if __name__ == "__main__":
    # Определяем корневую директорию (2 уровня вверх от текущего файла)
    current_dir = Path(__file__).parent
    root_dir = current_dir.parent.parent

    # Добавляем в начало пути, чтобы иметь приоритет над системными пакетами
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    # Выводим информацию для отладки (можно закомментировать в продакшене)
    print(f"[DEBUG] Добавлена корневая директория в sys.path: {root_dir}")
    print(f"[DEBUG] Текущий sys.path: {sys.path}")


def main():
    """Основная функция запуска приложения"""
    try:
        # Теперь импортируем после настройки пути
        from config.dependency_injector import DependencyInjector
        from core.exceptions import FaceDetectionError, FileHandlingError
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
        print("  python -m application.cli.run ./photos")
        print("  python -m application.cli.run ./photos ./results")
        return 1

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) == 3 else None

    try:
        injector = DependencyInjector()
        service = injector.get_face_clustering_service()
        return 0 if service.process_images(input_path, output_dir) else 1
    except FileHandlingError as e:
        print(f"Ошибка работы с файлами: {str(e)}")
        return 3
    except FaceDetectionError as e:
        print(f"Ошибка детекции лиц: {str(e)}")
        return 4
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        import traceback
        print(f"Детали ошибки:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())