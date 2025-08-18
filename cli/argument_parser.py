"""
Модуль для обработки аргументов командной строки и настройки окружения.
Содержит общий код для всех CLI-скриптов, избегая дублирования.
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional


def setup_project_environment() -> Tuple[Path, bool]:
    """
    Определяет корневую директорию проекта и добавляет её в sys.path.
    Возвращает корневую директорию и флаг успешности настройки.

    Returns:
        Tuple[Path, bool]: корневая директория и флаг успешности
    """
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent

    # Ищем корневую директорию, поднимаясь вверх до тех пор, пока не найдем признак проекта
    root_dir = current_dir
    while True:
        # Проверяем наличие признаков корневой директории проекта
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
        sys.path.insert(0, str(root_dir))
        print(f"[DEBUG] Добавлена корневая директория в sys.path: {root_dir}")
        return root_dir, True
    else:
        print(f"[DEBUG] Корневая директория уже в sys.path: {root_dir}")
        return root_dir, False


def create_argument_parser(description: str) -> argparse.ArgumentParser:
    """
    Создает стандартный парсер аргументов для CLI-приложений.

    Args:
        description: Описание приложения для отображения в справке

    Returns:
        argparse.ArgumentParser: Настроенный парсер аргументов
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', '--src', required=True, help='Путь к директории с изображениями')
    parser.add_argument('-o', '--output', default='groups.json', help='Путь к выходному JSON файлу')
    parser.add_argument('-d', '--dest', help='Директория для организации файлов по группам')
    parser.add_argument('-m', '--show-matrix', action='store_true', help='Отображать матрицу схожести в консоли')
    parser.add_argument('-r', '--references', action='store_true',
                        help="Отображать таблицу сопоставления с эталонами (файлы, начинающиеся с 'refer_')")
    return parser


def validate_paths(args: argparse.Namespace) -> bool:
    """
    Проверяет существование и корректность указанных путей.

    Args:
        args: Аргументы командной строки

    Returns:
        bool: True, если пути валидны, иначе False
    """
    if not args.src:
        print("Ошибка: Не указан источник (-s/--src)")
        return False

    if not Path(args.src).exists():
        print(f"Ошибка: Директория '{args.src}' не существует")
        return False

    if not Path(args.src).is_dir():
        print(f"Ошибка: '{args.src}' не является директорией")
        return False

    if args.dest:
        try:
            Path(args.dest).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Ошибка при работе с директорией назначения '{args.dest}': {e}")
            return False

    return True