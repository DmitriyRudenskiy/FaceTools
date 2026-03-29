import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics.models.sam import SAM3SemanticPredictor


# --- Конфигурация ---

@dataclass
class Settings:
    """Настройки приложения."""
    model_path: str = "models/sam3.pt"
    prompt: str = "person"
    pad_percent: float = 0.0
    background_color: Tuple[int, int, int] = (247, 247, 247)
    jpeg_quality: int = 95
    device: str = "auto"
    use_mask: bool = False  # False = вырезать квадратом, True = по маске
    transparent_background: bool = False  # Новое поле: сохранять ли прозрачный фон (PNG)


# --- Геометрия ---

@dataclass
class BBox:
    """Класс для представления ограничивающего прямоугольника."""
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)


def get_mask_bbox(mask: np.ndarray) -> Optional[BBox]:
    """Находит ограничивающий прямоугольник для маски."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return BBox(xmin=int(xmin), ymin=int(ymin), xmax=int(xmax), ymax=int(ymax))


# --- Модель ---

class SegmentationModel:
    """Обертка для SAM3 модели. Инициализируется один раз."""

    def __init__(self, checkpoint_path: str, device_str: str = "auto"):
        self.device = self._resolve_device(device_str)
        self.predictor = self._load_model(checkpoint_path)
        self.logger = logging.getLogger(__name__)

    def _resolve_device(self, device_str: str) -> str:
        if device_str != "auto":
            return device_str
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self, checkpoint_path: str) -> SAM3SemanticPredictor:
        overrides = {
            'conf': 0.25,
            'task': 'segment',
            'mode': 'predict',
            'imgsz': 644,
            'save': False,
            'half': False,
            'verbose': False,
            'model': checkpoint_path,
            'device': self.device
        }
        try:
            return SAM3SemanticPredictor(overrides=overrides)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {e}")

    def predict_masks(self, image_path: str, prompt: str) -> List[np.ndarray]:
        """Предсказывает маски для всех найденных объектов."""
        try:
            self.predictor.set_image(image_path)
            results = self.predictor(text=[prompt])

            if not results or results[0].masks is None:
                return []

            masks = []
            for mask_data in results[0].masks.data:
                mask = mask_data.cpu().numpy().astype(bool)
                if np.any(mask):
                    masks.append(mask)
            return masks

        except Exception as e:
            self.logger.error(f"Ошибка предсказания для {image_path}: {e}")
            return []


# --- Обработка изображений ---

def create_debug_image(
        image: Image.Image,
        masks: List[np.ndarray],
        bboxes: List[BBox]
) -> Image.Image:
    """Создает отладочное изображение."""
    debug_img = image.copy().convert("RGBA")
    colors = [(255, 0, 0, 80), (0, 255, 0, 80), (0, 0, 255, 80)]

    for i, (mask, bbox) in enumerate(zip(masks, bboxes)):
        color = colors[i % len(colors)]
        overlay = Image.new('RGBA', image.size, color)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        debug_img.paste(overlay, (0, 0), mask_pil)

        draw = ImageDraw.Draw(debug_img)
        draw.rectangle(bbox.to_tuple(), outline="red", width=2)
        draw.text((bbox.xmin, bbox.ymin - 10), f"Person {i + 1}", fill="red")

    return debug_img.convert("RGB")


def crop_person_to_square(
        image: Image.Image,
        mask: np.ndarray,
        settings: Settings
) -> Optional[Image.Image]:
    """
    Вырезает объект и вписывает его в квадрат.
    Поддерживает режимы:
    1. Прозрачный фон (PNG) - если settings.transparent_background = True.
    2. Заливка цветом (JPG) - если settings.transparent_background = False.
    """
    mask_bbox = get_mask_bbox(mask)
    if mask_bbox is None:
        return None

    # 1. Определяем координаты прямоугольника с учетом отступов (padding)
    pad_w = int(mask_bbox.width * settings.pad_percent / 100.0)
    pad_h = int(mask_bbox.height * settings.pad_percent / 100.0)

    # Расширяем границы
    xmin = mask_bbox.xmin - pad_w
    ymin = mask_bbox.ymin - pad_h
    xmax = mask_bbox.xmax + pad_w
    ymax = mask_bbox.ymax + pad_h

    # Размеры расширенного прямоугольника
    rect_w = xmax - xmin
    rect_h = ymax - ymin

    # 2. Размер итогового квадрата (по большей стороне прямоугольника)
    square_size = max(rect_w, rect_h)

    # 3. Создаем квадрат фона
    # Выбираем режим (RGBA для прозрачности, RGB для цвета) и цвет фона
    if settings.transparent_background:
        mode = "RGBA"
        bg_color = (0, 0, 0, 0)  # Полностью прозрачный
    else:
        mode = "RGB"
        bg_color = settings.background_color

    result = Image.new(mode, (square_size, square_size), bg_color)

    # 4. Вырезаем валидную часть из исходного изображения
    img_w, img_h = image.size

    src_xmin = max(0, xmin)
    src_ymin = max(0, ymin)
    src_xmax = min(img_w, xmax)
    src_ymax = min(img_h, ymax)

    if src_xmin >= src_xmax or src_ymin >= src_ymax:
        return result

    # Координаты для вставки
    center_x = square_size // 2
    center_y = square_size // 2
    ideal_center_x = (xmin + xmax) // 2
    ideal_center_y = (ymin + ymax) // 2

    paste_x = center_x - (ideal_center_x - src_xmin)
    paste_y = center_y - (ideal_center_y - src_ymin)

    # Вырезаем кусок оригинала
    cropped_rect = image.crop((src_xmin, src_ymin, src_xmax, src_ymax))

    # Если мы работаем в режиме RGBA, но исходник был RGB, конвертируем вырезанный кусок
    if mode == "RGBA" and cropped_rect.mode != "RGBA":
        cropped_rect = cropped_rect.convert("RGBA")

    # Логика вставки
    if settings.use_mask:
        # --- Режим вырезания по МАСКЕ ---
        # Вырезаем часть маски
        mask_crop = mask[src_ymin:src_ymax, src_xmin:src_xmax]
        mask_pil = Image.fromarray((mask_crop * 255).astype(np.uint8), mode='L')

        # Вставляем с маской.
        # Для RGB: маска определит, где фон, а где картинка.
        # Для RGBA: маска определит прозрачность (альфа-канал).
        result.paste(cropped_rect, (paste_x, paste_y), mask_pil)
    else:
        # --- Режим вырезания по КВАДРАТУ ---
        # Просто вставляем вырезанный прямоугольник.
        # Если результат RGBA, области вне картинки останутся прозрачными.
        result.paste(cropped_rect, (paste_x, paste_y))

    return result


# --- Основная логика ---

class PersonCropper:
    """Класс-оркестратор для обработки изображений."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.model = SegmentationModel(settings.model_path, settings.device)

    def process_image(self, image_path: str, save_debug: bool = False) -> int:
        """Обрабатывает одно изображение, возвращает количество найденных людей."""
        try:
            self.logger.info(f"Обработка файла: {image_path}")

            # Загружаем изображение
            image = Image.open(image_path)

            # Если входное изображение имеет палитру или транспарентность, конвертируем в RGB для корректной работы модели
            # Но сохраняем оригинал для вырезания, если нужен прозрачный фон
            if image.mode != 'RGB':
                # Создаем копию для модели
                image_for_model = image.convert('RGB')
            else:
                image_for_model = image

            # Если нужен прозрачный фон, лучше работать с RGBA версией исходника
            if self.settings.transparent_background and image.mode != 'RGBA':
                # Конвертируем исходник в RGBA, если он еще нет (например, JPG)
                image = image.convert('RGBA')

            masks = self.model.predict_masks(image_path, self.settings.prompt)

            if not masks:
                self.logger.warning(f"Люди не найдены: {image_path}")
                return 0

            self.logger.info(f"Найдено объектов: {len(masks)}")

            bboxes_for_debug = []
            saved_count = 0
            last_output_dir = None

            for i, mask in enumerate(masks):
                # Передаем исходное изображение (возможно RGBA)
                result_image = crop_person_to_square(image, mask, self.settings)

                if result_image:
                    # Определяем формат и расширение
                    if self.settings.transparent_background:
                        fmt = "PNG"
                        ext = ".png"
                    else:
                        fmt = "JPEG"
                        ext = ".jpg"

                    output_path = self._get_output_path(image_path, i if len(masks) > 1 else -1, ext)
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Сохраняем
                    save_kwargs = {}
                    if fmt == "JPEG":
                        save_kwargs['quality'] = self.settings.jpeg_quality
                        save_kwargs['subsampling'] = 0
                    # Для PNG качество обычно не настраивается через quality, можно добавить compress_level, но default ок.

                    result_image.save(str(output_path), format=fmt, **save_kwargs)

                    saved_count += 1
                    last_output_dir = output_path.parent
                    bboxes_for_debug.append(get_mask_bbox(mask))

            if save_debug and bboxes_for_debug and last_output_dir:
                # Для дебага всегда используем RGB
                self._save_debug(image_for_model, masks, bboxes_for_debug,
                                 last_output_dir / f"debug_{Path(image_path).name}")

            return saved_count

        except Exception as e:
            self.logger.error(f"Ошибка при обработке {image_path}: {e}", exc_info=True)
            return 0

    def _save_debug(self, image, masks, bboxes, output_path):
        debug_img = create_debug_image(image, masks, bboxes)
        debug_img.save(output_path)

    @staticmethod
    def _get_output_path(input_path: str, index: int = -1, ext: str = ".jpg") -> Path:
        input_p = Path(input_path)
        suffix = f"_{index}" if index >= 0 else ""
        return input_p.parent / f"{input_p.stem}_person{suffix}{ext}"


# --- Точка входа ---

def get_image_paths(input_path: str) -> List[str]:
    """
    Возвращает список путей к изображениям.
    Если папка — сканирует рекурсивно (rglob).
    """
    path_obj = Path(input_path)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    if path_obj.is_file():
        if path_obj.suffix.lower() in extensions:
            return [str(path_obj)]
        else:
            logging.error(f"Файл не является изображением: {input_path}")
            return []

    if path_obj.is_dir():
        return [str(p) for p in path_obj.rglob('*') if p.suffix.lower() in extensions]

    logging.error(f"Путь не найден: {input_path}")
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Вырезание людей и вписывание их в квадрат. Поддержка прозрачного фона (PNG)."
    )
    parser.add_argument("input_path", type=str, help="Путь к изображению или папке")
    parser.add_argument("--debug", action="store_true", help="Сохранить дебаг изображение")
    parser.add_argument("--verbose", action="store_true", help="Подробное логирование")

    parser.add_argument(
        "--mask",
        action="store_true",
        help="Вырезать по точной маске (удалить фон). По умолчанию вырезается прямоугольник."
    )

    # Новый аргумент
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Сохранить результат в PNG с прозрачным фоном. Работает автоматически с --mask."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 1. Определение путей
    script_dir = Path(__file__).parent.resolve()
    model_path = script_dir.parent / "models" / "sam3.pt"

    if not model_path.exists():
        logger.error(f"Файл модели не найден: {model_path}")
        sys.exit(1)

    # 2. Поиск изображений
    image_paths = get_image_paths(args.input_path)

    if not image_paths:
        logger.error("Изображения для обработки не найдены.")
        sys.exit(0)

    logger.info(f"Найдено изображений: {len(image_paths)}")

    # 3. Инициализация
    try:
        settings = Settings(
            model_path=str(model_path),
            use_mask=args.mask,
            transparent_background=args.transparent
        )
        cropper = PersonCropper(settings)
    except Exception as e:
        logger.error(f"Не удалось инициализировать модель: {e}")
        sys.exit(1)

    # 4. Цикл обработки
    total_processed = 0
    for img_path in image_paths:
        count = cropper.process_image(img_path, args.debug)
        total_processed += count

    logger.info(f"Обработка завершена. Всего вырезано людей: {total_processed}")


if __name__ == "__main__":
    main()