import argparse
import sys
import os
import numpy as np
import torch
from PIL import Image

# Импорты из Ultralytics (SAM 3)
from ultralytics.models.sam import SAM3SemanticPredictor


def get_bounding_box(mask):
    """Находит координаты (ymin, xmin, ymax, xmax) маски."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return ymin, xmin, ymax, xmax


def main():
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description="Скрипт для вырезания головы квадратным кропом (Ultralytics SAM3).")
    parser.add_argument("image_path", type=str, help="Путь к исходному изображению.")
    parser.add_argument("--output", "-o", type=str, default="cropped_head_square.jpg",
                        help="Путь для сохранения результата.")
    parser.add_argument("--pad", type=float, default=50.0, help="Отступ в процентах от самой большой стороны головы.")
    parser.add_argument("--prompt", type=str, default="head",
                        help="Текстовый запрос для поиска (например, 'head', 'person').")

    args = parser.parse_args()

    # --- РАСЧЕТ ПУТИ К ВЕСАМ ОТНОСИТЕЛЬНО СКРИПТА ---
    # Получаем абсолютный путь к папке, где лежит этот файл скрипта
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Формируем полный путь к ../models/sam3.pt относительно папки скрипта
    checkpoint_path = os.path.normpath(os.path.join(script_dir, "..", "models", "sam3.pt"))

    # Проверка на наличие файла
    if not os.path.exists(checkpoint_path):
        print(f"ОШИБКА: Файл весов не найден по абсолютному пути: {checkpoint_path}")
        print("Пожалуйста, скачайте sam3.pt и положите его в папку models на уровень выше скрипта.")
        sys.exit(1)

    # --- АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ УСТРОЙСТВА ---
    if torch.cuda.is_available():
        device_str = "0"  # CUDA
        device_name = "CUDA"
    elif torch.backends.mps.is_available():
        device_str = "mps"  # Apple Silicon
        device_name = "MPS (Apple Silicon)"
    else:
        device_str = "cpu"
        device_name = "CPU"

    print(f"Используемое устройство: {device_name}")

    # 1. Инициализация модели через Ultralytics
    print(f"Загрузка модели из: {checkpoint_path}...")
    try:
        # Настройка параметров предиктора
        overrides = dict(
            conf=0.25,  # Порог уверенности
            task="segment",  # Задача сегментации
            mode="predict",  # Режим предсказания
            model=checkpoint_path,  # Абсолютный путь к весам
            device=device_str,  # Устройство (cuda, mps, cpu)
            save=False,  # Не сохранять автоматически
            half=False  # Отключаем half precision для стабильности на Mac
        )

        predictor = SAM3SemanticPredictor(overrides=overrides)

    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        sys.exit(1)

    # 2. Загрузка изображения и предсказание
    print(f"Загрузка изображения: {args.image_path}")
    try:
        predictor.set_image(args.image_path)

        # Отправляем текстовый промпт
        results = predictor(text=[args.prompt])

        if not results or results[0].masks is None:
            print(f"Маски не найдены для промпта: '{args.prompt}'")
            return

        # Получаем маски (первую найденную)
        masks_tensor = results[0].masks.data
        mask = masks_tensor[0].cpu().numpy()

    except Exception as e:
        print(f"Ошибка при обработке изображений: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Логика квадратного кропа
    bbox = get_bounding_box(mask)

    if bbox is None:
        print("Не удалось определить границы объекта на маске.")
        return

    # Открываем изображение
    image_pil = Image.open(args.image_path).convert("RGB")
    img_w, img_h = image_pil.size

    ymin, xmin, ymax, xmax = bbox
    head_width = xmax - xmin
    head_height = ymax - ymin

    max_side = max(head_width, head_height)
    pad_pixels = int(max_side * (args.pad / 100.0))
    crop_size = max_side + (2 * pad_pixels)

    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    half_crop = crop_size / 2
    new_xmin = int(center_x - half_crop)
    new_xmax = int(center_x + half_crop)
    new_ymin = int(center_y - half_crop)
    new_ymax = int(center_y + half_crop)

    # Корректировка границ (чтобы не вылезти за фото)
    if new_xmax > img_w:
        diff = new_xmax - img_w
        new_xmax -= diff
        new_xmin -= diff
    if new_xmin < 0:
        diff = -new_xmin
        new_xmin += diff
        new_xmax += diff
    if new_ymax > img_h:
        diff = new_ymax - img_h
        new_ymax -= diff
        new_ymin -= diff
    if new_ymin < 0:
        diff = -new_ymin
        new_ymin += diff
        new_ymax += diff

    new_xmin = max(0, new_xmin)
    new_xmax = min(img_w, new_xmax)
    new_ymin = max(0, new_ymin)
    new_ymax = min(img_h, new_ymax)

    # Центрирование финального квадрата
    final_w = new_xmax - new_xmin
    final_h = new_ymax - new_ymin
    min_dim = min(final_w, final_h)

    if final_w > final_h:
        excess = final_w - min_dim
        new_xmin += excess // 2
        new_xmax -= excess - (excess // 2)
    elif final_h > final_w:
        excess = final_h - min_dim
        new_ymin += excess // 2
        new_ymax -= excess - (excess // 2)

    # 4. Сохранение
    cropped_img = image_pil.crop((new_xmin, new_ymin, new_xmax, new_ymax))
    cropped_img.save(args.output)
    print(f"Готово! Результат сохранен в: {args.output}")


if __name__ == "__main__":
    main()