#!/usr/bin/env python3
"""Точка входа CLI-приложения для кластеризации лиц.
Использование:
python -m application.cli.run <input_path> [output_dir]
python /path/to/extract_faces.py <input_path> [output_dir]
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path при прямом запуске
if __name__ == "__main__":
    # Определяем текущий файл и пытаемся найти корневую директорию проекта
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent

    # Ищем корневую директорию, поднимаясь вверх до тех пор, пока не найдем признак проекта
    root_dir = current_dir
    while True:
        # Проверяем наличие признаков корневой директории проекта
        has_src = (root_dir / "src").is_dir()

        # Проверяем наличие подкаталогов внутри src
        if has_src:
            src_dir = root_dir / "src"
            has_core = (src_dir / "core").is_dir()
            has_domain = (src_dir / "domain").is_dir()
            has_infrastructure = (src_dir / "infrastructure").is_dir()
            if has_core and has_domain and has_infrastructure:
                break

        # Поднимаемся на уровень выше
        parent_dir = root_dir.parent
        if parent_dir == root_dir:
            # Достигли корня файловой системы
            break
        root_dir = parent_dir

    # Добавляем корневую директорию в sys.path, если она найдена
    if str(root_dir) not in sys.path:
        current_file = Path(__file__).resolve()
        root_dir = current_file.parent.parent  # Это должно быть FaceCluster/
        sys.path.insert(0, str(root_dir))
        print(f"[DEBUG] Добавлена корневая директория в sys.path: {root_dir}")
    else:
        print(f"[DEBUG] Корневая директория уже в sys.path: {root_dir}")

    # Выводим текущий sys.path для отладки
    print("[DEBUG] Текущий sys.path:")
    for path in sys.path:
        print(f" {path}")


def main():
    """Основная функция запуска приложения"""
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
        # service теперь экземпляр FaceExtractionService
        service = injector.get_face_clustering_service()
        return 0 if service.process_images(input_path, output_dir) else 1
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        import traceback
        print(f"Детали ошибки:{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())