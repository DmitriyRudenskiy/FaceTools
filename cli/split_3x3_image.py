import os
import sys
from PIL import Image


def split_image_into_grid(image_path, output_dir):
    """
    Разрезает изображение на сетку 3x3 (9 частей).
    """
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Ошибка при открытии изображения {image_path}: {e}")
        return

    # Имя файла без расширения
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]

    # Создаем папку для вывода, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    width, height = img.size

    # Вычисляем размеры одной части
    # Предполагаем, что изображение квадратное или близко к тому
    # Делим на 3 части
    part_w = width // 3
    part_h = height // 3

    if part_w == 0 or part_h == 0:
        print(f"Изображение {image_path} слишком маленькое для нарезки.")
        return

    count = 1
    # Проходим по сетке 3x3
    # top, bottom, left, right
    for row in range(3):
        for col in range(3):
            left = col * part_w
            upper = row * part_h
            right = left + part_w
            lower = upper + part_h

            # Обрезаем изображение (crop)
            # Для последнего столбца/строки берем остаток до конца, чтобы не потерять пиксели при округлении
            if col == 2: right = width
            if row == 2: lower = height

            box = (left, upper, right, lower)
            cropped_img = img.crop(box)

            # Формируем имя файла: name_1.png, name_2.png ...
            output_filename = f"{base_name}_{count}{ext}"
            save_path = os.path.join(output_dir, output_filename)

            try:
                cropped_img.save(save_path)
                print(f"Сохранено: {save_path}")
            except Exception as e:
                print(f"Ошибка при сохранении {save_path}: {e}")

            count += 1


def main():
    # Проверка аргументов командной строки
    if len(sys.argv) < 2:
        print("Использование: python split_image.py <путь_к_изображению_или_папке>")
        print("Пример файла: python split_image.py image.png")
        print("Пример папки: python split_image.py ./images/")
        sys.exit(1)

    input_path = sys.argv[1]

    # Папка для сохранения результатов
    output_folder = "split_output"

    if os.path.isfile(input_path):
        # Если передан один файл
        print(f"Обработка файла: {input_path}")
        split_image_into_grid(input_path, output_folder)

    elif os.path.isdir(input_path):
        # Если передана папка
        print(f"Обработка папки: {input_path}")
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
        files = [f for f in os.listdir(input_path) if f.lower().endswith(valid_extensions)]

        if not files:
            print("В папке не найдено изображений.")
            return

        for filename in files:
            full_path = os.path.join(input_path, filename)
            split_image_into_grid(full_path, output_folder)

    else:
        print(f"Путь '{input_path}' не найден.")


if __name__ == "__main__":
    main()