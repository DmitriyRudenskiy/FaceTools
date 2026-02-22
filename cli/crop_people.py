import argparse
import logging
import sys
from dataclasses import dataclass
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
    background_color: Tuple[int, int, int] = (255, 255, 255)  # Белый цвет
    jpeg_quality: int = 95
    device: str = "auto"


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

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.xmin + self.xmax) // 2, (self.ymin + self.ymax) // 2)

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
    """Обертка для SAM3 модели."""

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
    """Создает отладочное изображение с визуализацией всех масок."""
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
    1. Вырезает прямоугольник с человеком (используя BBox маски).
    2. Создает квадрат размером с большую сторону прямоугольника.
    3. Вставляет прямоугольник в центр квадрата (без вырезания по маске).
    """
    mask_bbox = get_mask_bbox(mask)
    if mask_bbox is None:
        return None

    # 1. Определяем координаты прямоугольника с учетом отступов
    # Добавляем отступы (padding) к координатам
    pad_w = int(mask_bbox.width * settings.pad_percent / 100.0)
    pad_h = int(mask_bbox.height * settings.pad_percent / 100.0)

    xmin = mask_bbox.xmin - pad_w
    ymin = mask_bbox.ymin - pad_h
    xmax = mask_bbox.xmax + pad_w
    ymax = mask_bbox.ymax + pad_h

    # 2. Вычисляем размер будущего квадрата (по большей стороне прямоугольника)
    rect_w = xmax - xmin
    rect_h = ymax - ymin
    square_size = max(rect_w, rect_h)

    # 3. Создаем белый квадрат
    result = Image.new("RGB", (square_size, square_size), settings.background_color)

    # 4. Вычисляем область для вырезания из исходного изображения
    # Важно: координаты могут выходить за пределы изображения,
    # поэтому вырезаем только пересечение с оригиналом
    img_w, img_h = image.size

    # Область, которую хотим вырезать (может выходить за границы)
    src_xmin = max(0, xmin)
    src_ymin = max(0, ymin)
    src_xmax = min(img_w, xmax)
    src_ymax = min(img_h, ymax)

    # Если после отсечения область пустая
    if src_xmin >= src_xmax or src_ymin >= src_ymax:
        return result

    # Вырезаем кусок из оригинала
    cropped_rect = image.crop((src_xmin, src_ymin, src_xmax, src_ymax))

    # 5. Вычисляем, куда вклеить этот кусок на белом квадрате
    # Центрируем прямоугольник в квадрате.
    # Сначала находим центр квадрата
    center_x = square_size // 2
    center_y = square_size // 2

    # Смещение относительно центра, чтобы понять левый верхний угол
    paste_x = center_x - (rect_w // 2)
    paste_y = center_y - (rect_h // 2)

    # Корректируем смещение, если исходный прямоугольник выходил за границы изображения
    # Например, если xmin был < 0, то при вырезании мы потеряли левую пустую часть.
    # Значит вставлять нужно правее на эту разницу.
    paste_x += (src_xmin - xmin)
    paste_y += (src_ymin - ymin)

    # 6. Вставляем вырезанный прямоугольник на белый квадрат
    # БЕЗ использования маски (просто paste)
    result.paste(cropped_rect, (paste_x, paste_y))

    return result


# --- Основная логика ---

class PersonCropper:
    """Класс-оркестратор для обработки изображений."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.model = SegmentationModel(settings.model_path, settings.device)

    def process(self, image_path: str, save_debug: bool = False) -> int:
        try:
            self.logger.info(f"Обработка изображения: {image_path}")

            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            masks = self.model.predict_masks(image_path, self.settings.prompt)

            if not masks:
                self.logger.warning(f"Люди не найдены на изображении: {image_path}")
                return 0

            self.logger.info(f"Найдено объектов: {len(masks)}")

            bboxes_for_debug = []
            saved_count = 0

            for i, mask in enumerate(masks):
                result_image = crop_person_to_square(image, mask, self.settings)

                if result_image:
                    output_path = self._get_output_path(image_path, i if len(masks) > 1 else -1)
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    result_image.save(
                        str(output_path),
                        format="JPEG",
                        quality=self.settings.jpeg_quality,
                        subsampling=0
                    )
                    saved_count += 1
                    bboxes_for_debug.append(get_mask_bbox(mask))

            if save_debug and bboxes_for_debug:
                self._save_debug(image, masks, bboxes_for_debug, output_path.parent / f"debug_{Path(image_path).name}")

            return saved_count

        except Exception as e:
            self.logger.error(f"Ошибка при обработке {image_path}: {e}", exc_info=True)
            return 0

    def _save_debug(self, image, masks, bboxes, output_path):
        debug_img = create_debug_image(image, masks, bboxes)
        debug_img.save(output_path)

    @staticmethod
    def _get_output_path(input_path: str, index: int = -1) -> Path:
        input_p = Path(input_path)
        suffix = f"_{index}" if index >= 0 else ""
        return input_p.parent / f"{input_p.stem}_person{suffix}.jpg"


# --- Точка входа ---

def get_image_paths(input_path: str) -> List[str]:
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
        description="Вырезание людей и вписывание их в квадрат с белым фоном."
    )
    parser.add_argument("input_path", type=str, help="Путь к изображению или папке")
    parser.add_argument("--debug", action="store_true", help="Сохранить дебаг изображение")
    parser.add_argument("--verbose", action="store_true", help="Подробное логирование")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    script_dir = Path(__file__).parent.resolve()
    model_path = script_dir.parent / "models" / "sam3.pt"

    if not model_path.exists():
        logger.error(f"Файл модели не найден: {model_path}")
        sys.exit(1)

    image_paths = get_image_paths(args.input_path)

    if not image_paths:
        logger.error("Изображения для обработки не найдены.")
        sys.exit(1)

    logger.info(f"Найдено изображений: {len(image_paths)}")

    try:
        settings = Settings(model_path=str(model_path))
        cropper = PersonCropper(settings)

        total_processed = 0
        for img_path in image_paths:
            count = cropper.process(img_path, args.debug)
            total_processed += count

        logger.info(f"Обработка завершена. Всего вырезано людей: {total_processed}")

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()