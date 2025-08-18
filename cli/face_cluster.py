#!/usr/bin/env python3
"""
Инструмент для кластеризации лиц по схожести.
Использование: python -m application.cli.face_cluster <source_dir> [options]
"""
import sys
import argparse
from pathlib import Path

# Улучшенная обработка добавления корневой директории в sys.path
if __name__ == "__main__":
    # Определяем текущий файл и пытаемся найти корневую директорию проекта
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent

    # Ищем корневую директорию, поднимаясь вверх до тех пор, пока не найдем признак проекта
    root_dir = current_dir
    while True:
        # Проверяем наличие признаков корневой директории проекта
        has_core = (root_dir / "core").is_dir()
        has_domain = (root_dir / "domain").is_dir()
        has_infrastructure = (root_dir / "infrastructure").is_dir()

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
        sys.path.insert(0, str(root_dir))
        print(f"[DEBUG] Добавлена корневая директория в sys.path: {root_dir}")
    else:
        print(f"[DEBUG] Корневая директория уже в sys.path: {root_dir}")

    # Выводим текущий sys.path для отладки
    print("[DEBUG] Текущий sys.path:")
    for path in sys.path:
        print(f"  {path}")


def main():
    """Основная функция для кластеризации лиц"""
    parser = argparse.ArgumentParser(description='Кластеризация изображений по схожести лиц')
    parser.add_argument('-s', '--src', required=True, help='Путь к директории с изображениями')
    parser.add_argument('-o', '--output', default='groups.json', help='Путь к выходному JSON файлу')
    parser.add_argument('-d', '--dest', help='Директория для организации файлов по группам')
    parser.add_argument('-r', '--references', action='store_true', help='Отображать таблицу сопоставления с эталонами')

    args = parser.parse_args()

    try:
        # Импортируем после настройки пути
        from src.config.dependency_injector import DependencyInjector
    except ImportError as e:
        print(f"[ERROR] Не удалось импортировать config.dependency_injector: {str(e)}")
        print("Возможные причины:")
        print("1. Неправильно определена корневая директория проекта")
        print("2. Отсутствует файл __init__.py в корневой директории")
        print("3. Структура проекта не соответствует ожидаемой")
        return 1

    # Получаем сервис кластеризации
    try:
        injector = DependencyInjector()
        service = injector.get_face_clustering_service()
    except Exception as e:
        print(f"[ERROR] Ошибка инициализации сервиса: {str(e)}")
        return 1

    # Запускаем процесс
    try:
        success = service.process(args.src, args.output, args.dest)
    except Exception as e:
        print(f"[ERROR] Ошибка обработки изображений: {str(e)}")
        return 1

    # Дополнительная обработка для эталонов (если требуется)
    if args.references:
        print("\nТаблица сопоставления с эталонами (пока не реализовано в новой структуре)")
        # Здесь будет вызов специфичного сервиса для работы с эталонами

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())