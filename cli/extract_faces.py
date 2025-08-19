#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI-скрипт для извлечения лиц из изображений в указанной директории.
Сохраняет обрезанные лица в подкаталог 'faces' внутри исходной директории.
"""

import os
import sys
from pathlib import Path

# --- Настройка пути проекта ---
# Этот блок позволяет запускать скрипт из любого места,
# добавляя корневую директорию проекта в sys.path

# Определяем путь к текущему файлу и его родительские директории
current_file = Path(__file__).resolve()
current_dir = current_file.parent

# Ищем корневую директорию проекта (содержащую src, cli и т.д.)
# Поднимаемся вверх по дереву директорий
project_root = None
for parent in current_file.parents:
    # Проверяем наличие признаков корневой директории проекта
    src_dir = parent / "src"
    cli_dir = parent / "cli"
    # Можно добавить проверку на наличие файла __init__.py в корне или других признаков
    if src_dir.is_dir() and cli_dir.is_dir():
        project_root = parent
        break

if project_root is None:
    print("Ошибка: Не удалось определить корневую директорию проекта.")
    print("Убедитесь, что структура проекта корректна.")
    print("Текущий файл:", current_file)
    sys.exit(1)

# Добавляем корневую директорию в sys.path, если её там ещё нет
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

print(f"Корневая директория проекта определена как: {project_root}")

# --- Импорты после настройки пути ---
# Теперь можно безопасно импортировать модули проекта
try:
    from src.config.dependency_injector import DependencyInjector
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Возможные причины:")
    print("1. Неправильно определена корневая директория проекта")
    print("2. Отсутствует файл __init__.py в корневой директории")
    print("3. Структура проекта не соответствует ожидаемой")
    sys.exit(1)


def main():
    """
    Основная функция скрипта.
    """
    print("=== Скрипт извлечения лиц (extract_faces.py) ===")

    # Проверка аргументов командной строки
    if len(sys.argv) < 2:
        print("Использование: python extract_faces.py <путь_к_директории_с_изображениями> [путь_к_файлу_результатов.json]")
        print("Пример: python extract_faces.py '/path/to/images' 'faces_data.json'")
        return 1

    # Получаем пути из аргументов
    input_path = sys.argv[1]
    # Путь к файлу результатов - второй аргумент или None (сервис должен обработать)
    # Если нужно указать файл, можно сделать так:
    # output_file = sys.argv[2] if len(sys.argv) > 2 else "faces.json"
    # Но для extract_faces.py, как правило, это не нужно, поэтому передаем None
    # и ожидаем, что сервис корректно обработает этот случай.
    output_file = sys.argv[2] if len(sys.argv) > 2 else None # Или "faces.json" если хотите файл по умолчанию

    print(f"Входная директория: {input_path}")
    if output_file:
        print(f"Файл результатов: {output_file}")
    else:
        print("Файл результатов не будет создан (или будет создан с именем по умолчанию внутри сервиса).")

    # Проверка существования входной директории
    if not os.path.exists(input_path):
        print(f"Ошибка: Указанный путь не существует: {input_path}")
        return 1

    if not os.path.isdir(input_path):
        print(f"Ошибка: '{input_path}' не является директорией")
        return 1

    try:
        # Создаем инжектор зависимостей
        injector = DependencyInjector()

        # Получаем сервис для извлечения лиц
        # Предполагается, что этот сервис реализует process(input_path, output_file=None, dest_dir=None, ...)
        service = injector.get_face_crop_service() # Или get_face_processing_service, в зависимости от реализации

        # --- Исправление ---
        # Передаем output_file в метод process.
        # Если output_file == None, сервис должен корректно обработать этот случай
        # (например, не вызывать save или использовать путь по умолчанию внутри).
        # Также передаем dest_dir для организации файлов.
        dest_dir = os.path.join(input_path, "faces")
        print(f"Лица будут сохранены в: {dest_dir}")

        # Вызываем метод обработки изображений
        # Предполагаем, что метод process принимает output_file как аргумент
        # и корректно обрабатывает его значение None.
        success = service.process(
            input_path=input_path,
            output_file=output_file, # Передаем путь к файлу результатов (может быть None)
            dest_dir=dest_dir,       # Куда копировать обрезанные лица
            show_matrix=False,
            show_reference_table=False,
        )

        if success:
            print("Обработка изображений завершена успешно.")
            return 0 # Успешное завершение
        else:
            print("Ошибка при обработке изображений.")
            return 1 # Ошибка

    except Exception as e:
        print(f"[ERROR] Ошибка обработки изображений: {str(e)}")
        # Вывод трассировки стека для отладки (опционально)
        import traceback
        print(traceback.format_exc())
        return 1 # Ошибка


if __name__ == "__main__":
    # Запускаем основную функцию и выходим с соответствующим кодом
    sys.exit(main())
