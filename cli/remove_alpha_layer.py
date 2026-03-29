#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from PIL import Image


def replace_transparency_with_gray(input_path, output_path, gray_color=(247, 247, 247)):
    """
    Заменяет прозрачный фон на светло-серый и сохраняет изображение в JPG.
    """
    try:
        img = Image.open(input_path)

        # Конвертируем в RGBA, если нужно
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Создаем фон нужного цвета
        background = Image.new('RGB', img.size, gray_color)

        # Накладываем изображение с учетом альфа-канала
        background.paste(img, mask=img.split()[3])

        # Сохраняем в JPG
        background.save(output_path, 'JPEG', quality=95)
        return True

    except Exception as e:
        print(f"  ✗ Ошибка обработки {input_path}: {e}")
        return False


def process_directory(input_dir, output_dir=None, gray_color=(247, 247, 247)):
    """
    Обрабатывает все изображения в директории без рекурсии.
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"✗ Ошибка: Директория '{input_dir}' не найдена")
        return False

    if not input_path.is_dir():
        print(f"✗ Ошибка: '{input_dir}' не является директорией")
        return False

    # Определяем выходную директорию
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path

    # Поддерживаемые расширения
    supported_extensions = {'.png', '.PNG', '.webp', '.WEBP', '.gif', '.GIF', '.tiff', '.TIFF', '.bmp', '.BMP'}

    # Получаем список файлов (без рекурсии)
    files = [f for f in input_path.iterdir()
             if f.is_file() and f.suffix in supported_extensions]

    if not files:
        print(f"⚠ В директории не найдено поддерживаемых изображений")
        return True

    print(f"Найдено файлов для обработки: {len(files)}")
    print("-" * 50)

    success_count = 0
    fail_count = 0

    for file_path in files:
        # Формируем имя выходного файла
        output_file = output_path / f"{file_path.stem}.jpg"

        print(f"Обработка: {file_path.name} -> {output_file.name}")

        if replace_transparency_with_gray(file_path, output_file, gray_color):
            success_count += 1
            print(f"  ✓ Успешно")
        else:
            fail_count += 1

    print("-" * 50)
    print(f"✓ Обработано: {success_count}, Ошибок: {fail_count}")
    print(f"Результаты сохранены в: {output_path.absolute()}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Замена прозрачного фона на светло-серый для всех изображений в директории'
    )
    parser.add_argument('input_dir', help='Путь к входной директории с изображениями')
    parser.add_argument('output_dir', nargs='?', help='Путь к выходной директории (по умолчанию: входная)')
    parser.add_argument('--color', '-c', type=str, default='247,247,247',
                        help='Цвет фона в формате R,G,B (по умолчанию: 211,211,211)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Показать файлы для обработки без выполнения')

    args = parser.parse_args()

    # Парсим цвет
    try:
        gray_color = tuple(map(int, args.color.split(',')))
        if len(gray_color) != 3 or not all(0 <= c <= 255 for c in gray_color):
            raise ValueError
    except ValueError:
        print("✗ Ошибка: Цвет должен быть в формате R,G,B (значения 0-255)")
        exit(1)

    # Dry run
    if args.dry_run:
        input_path = Path(args.input_dir)
        supported_extensions = {'.png', '.PNG', '.webp', '.WEBP', '.gif', '.GIF', '.tiff', '.TIFF', '.bmp', '.BMP'}
        files = [f for f in input_path.iterdir()
                 if f.is_file() and f.suffix in supported_extensions]

        print(f"Файлы для обработки ({len(files)}):")
        for f in files:
            print(f"  {f.name} -> {f.stem}.jpg")
        return

    process_directory(args.input_dir, args.output_dir, gray_color)


if __name__ == '__main__':
    main()