import os
import sys
from PIL import Image

# Константы
CROP_SIZE = 720
CROP_PREFIX = "crop_"


def crop_images_in_directory(directory):
    # Проверяем, существует ли директория
    if not os.path.isdir(directory):
        print(f"Ошибка: Директория '{directory}' не найдена.")
        return

    # Создаем папку для сохранения обрезанных изображений
    output_dir = os.path.join(directory, "cropped")
    os.makedirs(output_dir, exist_ok=True)

    # Поддерживаемые форматы изображений
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

    # Проходим по всем файлам в директории
    for filename in os.listdir(directory):
        # Игнорируем файлы, которые уже начинаются с префикса crop_
        if filename.startswith(CROP_PREFIX):
            continue

        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(directory, filename)
            try:
                with Image.open(file_path) as img:
                    # Проверяем, что изображение достаточно большое
                    if img.width < CROP_SIZE or img.height < CROP_SIZE:
                        print(f"Пропущено (слишком маленькое): {filename}")
                        continue

                    # Обрезаем изображение до CROP_SIZE x CROP_SIZE, начиная с (0, 0)
                    cropped_img = img.crop((0, 0, CROP_SIZE, CROP_SIZE))

                    # Добавляем префикс к имени файла
                    name, ext = os.path.splitext(filename)
                    new_filename = f"{CROP_PREFIX}{name}{ext}"
                    output_path = os.path.join(output_dir, new_filename)

                    cropped_img.save(output_path)
                    print(f"Обрезано: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Ошибка при обработке файла '{filename}': {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python crop_images.py <путь_к_директории>")
    else:
        directory = sys.argv[1]
        crop_images_in_directory(directory)