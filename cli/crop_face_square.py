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
    prompt: str = "face"
    pad_percent: float = 0.0
    background_color: Tuple[int, int, int] = (220, 220, 220)
    jpeg_quality: int = 100
    device: str = "auto"  # auto, cpu, cuda, mps


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
        """Возвращает координаты как кортеж для PIL."""
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


def calculate_crop_bbox(
        mask_bbox: BBox,
        img_w: int,
        img_h: int,
        pad_percent: float
) -> BBox:
    """Вычисляет квадратную область для кропа, центрированную по объекту."""
    # 1. Определяем размер кропа
    head_size = max(mask_bbox.width, mask_bbox.height)
    pad_pixels = int(head_size * pad_percent / 100.0)
    crop_size = head_size + (2 * pad_pixels)

    # 2. Центрируем относительно маски
    center_x, center_y = mask_bbox.center
    half_crop = crop_size // 2

    xmin = center_x - half_crop
    xmax = center_x + half_crop
    ymin = center_y - half_crop
    ymax = center_y + half_crop

    # 3. Ограничиваем границами изображения (Clamp)
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w, xmax)
    ymax = min(img_h, ymax)

    # 4. Обеспечиваем квадратность, если после ограничений размеры изменились
    current_w = xmax - xmin
    current_h = ymax - ymin
    min_dim = min(current_w, current_h)

    # Пересчитываем центр, чтобы сохранить визуальный баланс внутри доступной области
    cx_clamped = (xmin + xmax) // 2
    cy_clamped = (ymin + ymax) // 2

    half_dim = min_dim // 2
    return BBox(
        xmin=cx_clamped - half_dim,
        ymin=cy_clamped - half_dim,
        xmax=cx_clamped + half_dim,
        ymax=cy_clamped + half_dim
    )


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

    def predict_mask(self, image_path: str, prompt: str) -> Optional[np.ndarray]:
        """Предсказывает маску для изображения."""
        try:
            self.predictor.set_image(image_path)
            results = self.predictor(text=[prompt])

            if not results or results[0].masks is None:
                return None

            mask = results[0].masks.data[0].cpu().numpy().astype(bool)
            return mask if np.any(mask) else None

        except Exception as e:
            self.logger.error(f"Ошибка предсказания для {image_path}: {e}")
            return None


# --- Обработка изображений ---

def create_debug_image(
        image: Image.Image,
        mask: np.ndarray,
        mask_bbox: BBox,
        crop_bbox: BBox
) -> Image.Image:
    """Создает отладочное изображение с визуализацией."""
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)

    # Желтый полупрозрачный оверлей для маски
    yellow_overlay = Image.new('RGBA', image.size, (255, 255, 0, 100))
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    debug_img.paste(yellow_overlay, (0, 0), mask_pil)

    # Рамки (вызываем метод to_tuple())
    draw.rectangle(mask_bbox.to_tuple(), outline="red", width=3)
    draw.rectangle(crop_bbox.to_tuple(), outline="blue", width=3)

    # Подписи
    draw.text((mask_bbox.xmin, mask_bbox.ymin - 20), "Mask BBox", fill="red")
    draw.text((crop_bbox.xmin, crop_bbox.ymin - 20), "Crop BBox", fill="blue")

    return debug_img


def simple_crop(
        image: Image.Image,
        crop_bbox: BBox
) -> Image.Image:
    """Просто вырезает квадратную область из изображения."""
    # Вырезаем квадратную область (вызываем метод to_tuple())
    cropped_img = image.crop(crop_bbox.to_tuple())
    return cropped_img


# --- Основная логика ---

class HeadCropper:
    """Класс-оркестратор для обработки изображений."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        # Модель загружается один раз при инициализации класса
        self.model = SegmentationModel(settings.model_path, settings.device)

    def process(self, image_path: str, save_debug: bool = False) -> bool:
        try:
            self.logger.info(f"Обработка изображения: {image_path}")

            # Загрузка
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Предсказание
            mask = self.model.predict_mask(image_path, self.settings.prompt)
            if mask is None:
                self.logger.warning(f"Не удалось найти голову на изображении: {image_path}")
                return False

            # Геометрия
            mask_bbox = get_mask_bbox(mask)
            if mask_bbox is None:
                self.logger.warning(f"Не удалось определить границы маски: {image_path}")
                return False

            # Вычисляем квадратную область для кропа
            crop_bbox = calculate_crop_bbox(
                mask_bbox, *image.size, self.settings.pad_percent
            )

            # Вырезаем квадрат
            result_image = simple_crop(image, crop_bbox)

            output_path = self._get_output_path(image_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_image.save(
                str(output_path),
                format="JPEG",
                quality=self.settings.jpeg_quality,
                subsampling=0
            )
            self.logger.info(f"Результат сохранен: {output_path}")

            if save_debug:
                self._save_debug(image, mask, mask_bbox, crop_bbox, output_path)

            return True

        except Exception as e:
            self.logger.error(f"Ошибка при обработке {image_path}: {e}", exc_info=True)
            return False

    def _save_debug(self, image, mask, mask_bbox, crop_bbox, output_path):
        debug_img = create_debug_image(image, mask, mask_bbox, crop_bbox)
        debug_path = output_path.parent / f"debug_{output_path.name}"
        debug_img.save(debug_path)
        self.logger.info(f"Отладочное изображение: {debug_path}")

    @staticmethod
    def _get_output_path(input_path: str) -> Path:
        input_p = Path(input_path)
        return input_p.parent / f"{input_p.stem}_face.jpg"


# --- Точка входа ---

def get_image_paths(input_path: str) -> List[str]:
    """Возвращает список путей к изображениям. Если папка — сканирует рекурсивно."""
    path_obj = Path(input_path)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    if path_obj.is_file():
        if path_obj.suffix.lower() in extensions:
            return [str(path_obj)]
        else:
            logging.error(f"Файл не является изображением: {input_path}")
            return []

    if path_obj.is_dir():
        # Ищем все изображения в подпапках
        return [str(p) for p in path_obj.rglob('*') if p.suffix.lower() in extensions]

    logging.error(f"Путь не найден: {input_path}")
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Вырезание квадратной области вокруг головы с использованием SAM3."
    )
    parser.add_argument("input_path", type=str, help="Путь к изображению или папке с изображениями")
    parser.add_argument("--debug", action="store_true", help="Сохранить дебаг изображение")
    parser.add_argument("--verbose", action="store_true", help="Подробное логирование")

    args = parser.parse_args()

    # Настройка логирования
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Определение путей
    script_dir = Path(__file__).parent.resolve()
    model_path = script_dir.parent / "models" / "sam3.pt"

    # Проверки
    if not model_path.exists():
        logger.error(f"Файл модели не найден: {model_path}")
        sys.exit(1)

    # Получаем список всех изображений для обработки
    image_paths = get_image_paths(args.input_path)

    if not image_paths:
        logger.error("Изображения для обработки не найдены.")
        sys.exit(1)

    logger.info(f"Найдено изображений: {len(image_paths)}")

    try:
        settings = Settings(model_path=str(model_path))

        # Инициализация процессора (и загрузка модели) происходит ОДИН РАЗ
        cropper = HeadCropper(settings)

        processed_count = 0
        failed_count = 0

        for img_path in image_paths:
            if cropper.process(img_path, args.debug):
                processed_count += 1
            else:
                failed_count += 1

        logger.info(f"Обработка завершена. Успешно: {processed_count}, Ошибок: {failed_count}")

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()