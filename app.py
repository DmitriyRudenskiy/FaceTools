import sys
import os
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import numpy as np

def calculate_iou(box1, box2):
    """
    Вычисляет IoU (Intersection over Union) двух bounding box'ов
    Args:
        box1, box2: Координаты bounding box'ов [x1, y1, x2, y2]
    Returns:
        IoU значение (от 0 до 1)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    # Координаты пересечения
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    # Проверяем, есть ли пересечение
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    # Площадь пересечения
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    # Площади bounding box'ов
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    # IoU
    iou = inter_area / (area1 + area2 - inter_area)
    return iou

def merge_overlapping_boxes(boxes, iou_threshold=0.5):
    """
    Объединяет пересекающиеся bounding box'ы
    Args:
        boxes: Список bounding box'ов
        iou_threshold: Порог IoU для объединения
    Returns:
        Список объединенных bounding box'ов
    """
    if len(boxes) <= 1:
        return boxes
    # Сортируем box'ы по площади (от больших к меньшим)
    boxes_with_areas = [(box, (box[2] - box[0]) * (box[3] - box[1])) for box in boxes]
    boxes_with_areas.sort(key=lambda x: x[1], reverse=True)
    sorted_boxes = [box for box, area in boxes_with_areas]
    merged_boxes = []
    used = [False] * len(sorted_boxes)
    for i, box1 in enumerate(sorted_boxes):
        if used[i]:
            continue
        # Находим все box'ы, которые пересекаются с текущим больше чем на threshold
        to_merge = [box1]
        used[i] = True
        for j, box2 in enumerate(sorted_boxes[i+1:], i+1):
            if used[j]:
                continue
            iou = calculate_iou(box1, box2)
            if iou > iou_threshold:
                to_merge.append(box2)
                used[j] = True
        # Объединяем все найденные box'ы в один (берем минимальные x1,y1 и максимальные x2,y2)
        if len(to_merge) > 1:
            min_x1 = min(box[0] for box in to_merge)
            min_y1 = min(box[1] for box in to_merge)
            max_x2 = max(box[2] for box in to_merge)
            max_y2 = max(box[3] for box in to_merge)
            merged_box = [min_x1, min_y1, max_x2, max_y2]
            merged_boxes.append(merged_box)
        else:
            merged_boxes.append(box1)
    return merged_boxes

### ОБНОВЛЕННАЯ ФУНКЦИЯ ###
def calculate_square_crop(bbox, image_size):
    """
    Рассчитывает координаты квадратной области для обрезки.
    Сверху не добавляется ничего, снизу 20%, слева и справа по 20%.
    Результат - квадратная область.
    Args:
        bbox: Координаты bounding box [x1, y1, x2, y2]
        image_size: Размер изображения (width, height)
    Returns:
        Кортеж с координатами квадрата (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = map(int, bbox)
    img_width, img_height = image_size

    # Вычисляем ширину и высоту bounding box
    width = x2 - x1
    height = y2 - y1

    # Определяем размер квадрата (по большей стороне)
    square_size = max(width, height)

    # --- ДОБАВЛЕНИЕ ПРОЦЕНТОВ ---
    # Вычисляем отступы: 20% для левого, правого края и низа
    padding_horizontal = int(0.20 * square_size) # Отступы слева и справа
    padding_bottom = int(0.20 * square_size)     # Отступ снизу
    # Сверху отступ не добавляется

    # --- ОПРЕДЕЛЕНИЕ РАЗМЕРОВ ИТОГОВОГО КВАДРАТА ---
    # Ширина итогового квадрата
    final_width = width + 2 * padding_horizontal
    # Высота итогового квадрата (берем исходную высоту + отступ снизу)
    # Чтобы сделать его квадратным, его высота должна быть равна ширине
    final_height = final_width

    # Проверим, достаточно ли у нас высоты от y1 до нижней границы изображения
    available_height_below = img_height - y1
    if final_height > available_height_below:
        # Если нет, уменьшаем высоту до доступной
        final_height = available_height_below
        # Пересчитываем ширину, чтобы она соответствовала новой высоте (квадрат)
        final_width = final_height

    # --- ВЫЧИСЛЕНИЕ КООРДИНАТ ---
    # Центр bounding box по X
    center_x = (x1 + x2) // 2

    # Левая граница квадрата (с учетом пересчета ширины, если она изменилась)
    half_final_width = final_width // 2
    new_x1 = center_x - half_final_width
    new_x2 = center_x + half_final_width

    # Верхняя граница квадрата (без отступа сверху)
    new_y1 = y1
    # Нижняя граница квадрата
    new_y2 = new_y1 + final_height

    # --- КОРРЕКЦИЯ ГРАНИЦ ИЗОБРАЖЕНИЯ ---
    # Корректируем по X
    if new_x1 < 0:
        offset = -new_x1
        new_x1 += offset
        new_x2 += offset
    if new_x2 > img_width:
        offset = new_x2 - img_width
        new_x1 -= offset
        new_x2 -= offset
        # Повторная проверка левой границы
        if new_x1 < 0:
            new_x1 = 0
            new_x2 = min(img_width, new_x1 + final_width) # Используем пересчитанную ширину

    # Корректируем по Y (в основном вниз)
    if new_y2 > img_height:
        offset = new_y2 - img_height
        new_y1 -= offset
        new_y2 -= offset
        # Повторная проверка верхней границы
        if new_y1 < 0:
            new_y1 = 0
            new_y2 = min(img_height, new_y1 + final_height) # Используем пересчитанную высоту

    # Финальная проверка границ
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_width, new_x2)
    new_y2 = min(img_height, new_y2)

    # Убедимся, что это квадрат
    # Если из-за ограничений изображения квадрат "не получился",
    # выбираем меньшую сторону как размер квадрата
    final_actual_size = min(new_x2 - new_x1, new_y2 - new_y1)
    # Корректируем координаты, чтобы получить квадрат
    # Центрируем по X
    center_x_actual = (new_x1 + new_x2) // 2
    new_x1 = center_x_actual - final_actual_size // 2
    new_x2 = new_x1 + final_actual_size
    # Корректируем по X, если вышло за границы
    if new_x1 < 0:
        new_x1 = 0
        new_x2 = final_actual_size
    if new_x2 > img_width:
        new_x2 = img_width
        new_x1 = new_x2 - final_actual_size

    # Центрируем по Y, но учитываем, что сверху не должно быть отступа,
    # поэтому сдвигаем вниз, если нужно
    # new_y2 = new_y1 + final_actual_size
    # Но если new_y1 (исходный y1) + final_actual_size > img_height,
    # то нужно сдвинуть вверх
    if new_y1 + final_actual_size > img_height:
         new_y2 = img_height
         new_y1 = new_y2 - final_actual_size
    else:
         new_y2 = new_y1 + final_actual_size

    # Финальная проверка
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_width, new_x2)
    new_y2 = min(img_height, new_y2)

    return (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
### КОНЕЦ ОБНОВЛЕННОЙ ФУНКЦИИ ###

def crop_and_save_faces(results, original_image, image_path, output_dir):
    """
    Вырезает найденные лица квадратами и сохраняет их в отдельные файлы
    Args:
        results: Результаты детекции
        original_image: Оригинальное изображение
        image_path: Путь к оригинальному файлу
        output_dir: Директория для сохранения результатов
    """
    # Извлекаем имя файла
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    # Вырезаем лица и сохраняем в отдельные файлы
    if len(results.xyxy) > 0:
        # Преобразуем bounding box'ы в список
        boxes = [bbox.tolist() for bbox in results.xyxy]
        # Объединяем пересекающиеся box'ы
        merged_boxes = merge_overlapping_boxes(boxes, iou_threshold=0.5)
        print(f"Найдено {len(results.xyxy)} лиц, объединено в {len(merged_boxes)} областей")
        for i, bbox in enumerate(merged_boxes):
            # Рассчитываем координаты квадратной области
            x1, y1, x2, y2 = calculate_square_crop(bbox, original_image.size)
            # Обрезаем квадратную область из оригинального изображения
            face_image = original_image.crop((x1, y1, x2, y2))
            # Создаем имя файла: оригинальное имя + порядковый номер + .jpg
            face_filename = f"{image_name}_face_{i+1}.jpg"
            face_filepath = os.path.join(output_dir, face_filename)
            # Сохраняем обрезанное лицо
            face_image.save(face_filepath)
            print(f"Сохранено лицо {i+1}: {face_filepath}")
    else:
        print("Лица не найдены")

def get_image_files(path):
    """
    Создает массив изображений из файла или директории
    Args:
        path: Путь к файлу или директории
    Returns:
        Список путей к графическим файлам
    """
    # Поддерживаемые форматы изображений
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    # Проверка существования пути
    if not os.path.exists(path):
        print(f"Ошибка: Путь '{path}' не существует")
        return []
    # Если это файл
    if os.path.isfile(path):
        _, ext = os.path.splitext(path.lower())
        if ext in image_extensions:
            return [path]
        else:
            print(f"Ошибка: Файл '{path}' не является графическим файлом")
            return []
    # Если это директория
    elif os.path.isdir(path):
        images = []
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename.lower())
                if ext in image_extensions:
                    images.append(file_path)
        if not images:
            print(f"Ошибка: В директории '{path}' не найдено графических файлов")
            return []
        return images
    else:
        print(f"Ошибка: Путь '{path}' не является ни файлом, ни директорией")
        return []

def main(image_path, output_dir=None):
    """
    Главный метод для обработки изображений
    Args:
        image_path: Путь к файлу или директории с изображениями
        output_dir: Директория для сохранения результатов (опционально)
    """
    print(f"Начинаем обработку: {image_path}")
    # Получаем список изображений
    images = get_image_files(image_path)
    if not images:
        print("Ошибка: Не найдено изображений для обработки")
        return False
    # Если директория для сохранения не указана, используем директорию исходного файла
    if output_dir is None:
        if os.path.isfile(image_path):
            output_dir = os.path.dirname(image_path)
        else:
            output_dir = image_path
    # Создаем директорию для сохранения, если она не существует
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Создана директория для сохранения: {output_dir}")
        except Exception as e:
            print(f"Ошибка создания директории {output_dir}: {e}")
            return False
    elif not os.path.isdir(output_dir):
        print(f"Ошибка: Путь {output_dir} не является директорией")
        return False
    print(f"Директория для сохранения результатов: {output_dir}")
    # Определяем путь к модели (в той же директории, что и скрипт)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.pt")
    # Проверяем существование файла модели
    if not os.path.exists(model_path):
        print(f"Ошибка: Файл модели '{model_path}' не найден")
        return False
    # load model
    try:
        model = YOLO(model_path)
        print(f"Модель загружена: {model_path}")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return False
    # Обрабатываем каждое изображение
    for img_path in images:
        print(f"\nОбработка файла: {img_path}")
        try:
            # Загрузка изображения
            original_image = Image.open(img_path)
            # Выполняем детекцию
            output = model(original_image)
            results = Detections.from_ultralytics(output[0])
            print(results)
            # Вызываем метод для вырезания и сохранения лиц
            crop_and_save_faces(results, original_image, img_path, output_dir)
        except Exception as e:
            print(f"Ошибка обработки файла {img_path}: {e}")
            continue
    print("\nОбработка завершена!")
    return True

if __name__ == "__main__":
    # Проверяем количество аргументов
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Использование: python app.py <путь_к_файлу_или_директории> [директория_для_сохранения]")
        print("Пример: python app.py /Users/user/Downloads/image.jpg")
        print("Пример: python app.py /Users/user/Downloads/images/")
        print("Пример: python app.py /Users/user/Downloads/image.jpg /Users/user/Results/")
        print("Пример: python app.py /Users/user/Downloads/images/ /Users/user/Results/")
        sys.exit(1)
    # Получаем пути из аргументов командной строки
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    # Вызываем главный метод
    success = main(image_path, output_dir)
    if not success:
        sys.exit(1)
