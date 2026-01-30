import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics.models.sam import SAM3SemanticPredictor


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


class Logger:
    """Утилита для логирования."""

    @staticmethod
    def setup(verbose: bool = False) -> None:
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


class DebugVisualizer:
    """Класс для создания отладочной визуализации."""

    @staticmethod
    def create_debug_image(
            image: Image.Image,
            mask: np.ndarray,
            mask_bbox: BBox,
            crop_bbox: BBox
    ) -> Image.Image:
        """Создает изображение с наложенной маской и рамками."""
        debug_img = image.copy()
        draw = ImageDraw.Draw(debug_img)

        # Создаем желтый полупрозрачный оверлей
        yellow_overlay = Image.new('RGBA', image.size, (255, 255, 0, 100))

        # Преобразуем маску NumPy в PIL Image
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        debug_img.paste(yellow_overlay, (0, 0), mask_pil)

        # Рисуем рамки
        draw.rectangle(mask_bbox.to_tuple(), outline="red", width=3)
        draw.rectangle(crop_bbox.to_tuple(), outline="blue", width=3)

        # Добавляем подписи
        draw.text((mask_bbox.xmin, mask_bbox.ymin - 20),
                  "Mask BBox", fill="red")
        draw.text((crop_bbox.xmin, crop_bbox.ymin - 20),
                  "Crop BBox", fill="blue")

        return debug_img


class ModelConfig:
    """Конфигурация модели."""
    DEFAULT_OVERRIDES = {
        'conf': 0.25,
        'task': 'segment',
        'mode': 'predict',
        'imgsz': 644,
        'save': False,
        'half': False,
        'verbose': False
    }


class HeadCropSAM3:
    """Класс для вырезания головы с использованием SAM3."""

    def __init__(self, checkpoint_path: str, device_str: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device_str = device_str
        self.predictor = None
        self._init_model()

    def _init_model(self) -> None:
        """Инициализирует модель SAM3."""
        overrides = ModelConfig.DEFAULT_OVERRIDES.copy()
        overrides.update({
            'model': self.checkpoint_path,
            'device': self.device_str,
        })

        try:
            self.predictor = SAM3SemanticPredictor(overrides=overrides)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {e}")

    @staticmethod
    def get_bounding_box(mask: np.ndarray) -> Optional[BBox]:
        """Находит ограничивающий прямоугольник для маски."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        return BBox(xmin=int(xmin), ymin=int(ymin),
                    xmax=int(xmax), ymax=int(ymax))

    def predict_head_mask(self, image_path: str, prompt: str = "face") -> Optional[np.ndarray]:
        """Предсказание маски головы."""
        try:
            self.predictor.set_image(image_path)
            results = self.predictor(text=[prompt])

            if not results or results[0].masks is None:
                return None

            mask = results[0].masks.data[0].cpu().numpy().astype(bool)
            return mask if np.any(mask) else None

        except Exception as e:
            logging.error(f"Ошибка предсказания: {e}")
            return None

    @staticmethod
    def calculate_crop_region(
            mask_bbox: BBox,
            image_size: Tuple[int, int],
            pad_percent: float = 0.0
    ) -> BBox:
        """Вычисляет квадратную область для кропа."""
        img_w, img_h = image_size

        # Определяем размер кропа на основе размера головы и отступа
        head_size = max(mask_bbox.width, mask_bbox.height)
        pad_pixels = int(head_size * pad_percent / 100.0)
        crop_size = head_size + (2 * pad_pixels)

        center_x, center_y = mask_bbox.center
        half_crop = crop_size // 2

        # Вычисляем начальные границы
        xmin = center_x - half_crop
        xmax = center_x + half_crop
        ymin = center_y - half_crop
        ymax = center_y + half_crop

        # Корректируем границы если выходим за пределы изображения
        bbox = BBox(xmin, ymin, xmax, ymax)
        bbox = HeadCropSAM3._adjust_bbox_to_bounds(bbox, img_w, img_h)

        # Обеспечиваем квадратность
        bbox = HeadCropSAM3._ensure_square_bbox(bbox)

        return bbox

    @staticmethod
    def _adjust_bbox_to_bounds(bbox: BBox, img_w: int, img_h: int) -> BBox:
        """Корректирует bbox чтобы оставаться в пределах изображения."""
        xmin = max(0, bbox.xmin)
        ymin = max(0, bbox.ymin)
        xmax = min(img_w, bbox.xmax)
        ymax = min(img_h, bbox.ymax)

        return BBox(xmin, ymin, xmax, ymax)

    @staticmethod
    def _ensure_square_bbox(bbox: BBox) -> BBox:
        """Обеспечивает квадратность bbox."""
        width = bbox.width
        height = bbox.height

        if width == height:
            return bbox

        min_dim = min(width, height)
        center_x, center_y = bbox.center

        half_dim = min_dim // 2
        return BBox(
            xmin=center_x - half_dim,
            ymin=center_y - half_dim,
            xmax=center_x + half_dim,
            ymax=center_y + half_dim
        )

    @staticmethod
    def get_device() -> Tuple[str, str]:
        """Определяет доступное устройство для вычислений."""
        if torch.cuda.is_available():
            return "cuda", "CUDA"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps", "MPS (Apple Silicon)"
        return "cpu", "CPU"


class ImageProcessor:
    """Обработчик изображений."""

    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        """Загружает и конвертирует изображение в RGB."""
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise IOError(f"Не удалось загрузить изображение {image_path}: {e}")

    @staticmethod
    def apply_mask_to_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Применяет маску к изображению, создавая прозрачный фон."""
        if image.mode != 'RGBA':
            image_rgba = image.convert('RGBA')
        else:
            image_rgba = image.copy()

        alpha_data = (mask * 255).astype(np.uint8)
        alpha_image = Image.fromarray(alpha_data, mode='L')
        image_rgba.putalpha(alpha_image)

        return image_rgba

    @staticmethod
    def create_background_with_crop(
            crop_bbox: BBox,
            cropped_image: Image.Image,
            background_color: Tuple[int, int, int] = (220, 220, 220)
    ) -> Image.Image:
        """Создает изображение с фоном и кропом."""
        background = Image.new(
            "RGB",
            (crop_bbox.width, crop_bbox.height),
            background_color
        )
        background.paste(cropped_image, mask=cropped_image)
        return background

    @staticmethod
    def save_image(
            image: Image.Image,
            output_path: Path,
            quality: int = 100
    ) -> None:
        """Сохраняет изображение с указанным качеством."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(output_path), format="JPEG", quality=quality, subsampling=0)


class HeadCropProcessor:
    """Основной процессор для вырезания головы."""

    def __init__(
            self,
            checkpoint_path: str,
            device: str = "cpu",
            pad_percent: float = 0.0,
            prompt: str = "face",
            background_color: Tuple[int, int, int] = (220, 220, 220)
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.pad_percent = pad_percent
        self.prompt = prompt
        self.background_color = background_color
        self.logger = Logger.get_logger(__name__)

        self.head_cropper = HeadCropSAM3(checkpoint_path, device)
        self.image_processor = ImageProcessor()

    def process(self, image_path: str, debug: bool = False) -> bool:
        """Основной метод обработки изображения."""
        try:
            self.logger.info(f"Обработка изображения: {image_path}")
            self.logger.info(f"Используется запрос: '{self.prompt}'")

            # Загрузка изображения
            image = self.image_processor.load_image(image_path)

            # Предсказание маски
            mask = self.head_cropper.predict_head_mask(image_path, self.prompt)
            if mask is None:
                self.logger.error("Не удалось найти голову на изображении")
                return False

            # Определение bounding box
            mask_bbox = self.head_cropper.get_bounding_box(mask)
            if mask_bbox is None:
                self.logger.error("Не удалось определить границы объекта")
                return False

            # Вычисление региона для кропа
            crop_bbox = self.head_cropper.calculate_crop_region(
                mask_bbox, image.size, self.pad_percent
            )

            # Вырезание и обработка изображения
            result_image = self._crop_and_process_image(image, mask, crop_bbox)

            # Сохранение результата
            output_path = self._get_output_path(image_path)
            self.image_processor.save_image(result_image, output_path)
            self.logger.info(f"Результат сохранен: {output_path}")

            # Отладочная визуализация
            if debug:
                self._save_debug_image(image, mask, mask_bbox, crop_bbox, output_path)

            return True

        except Exception as e:
            self.logger.error(f"Ошибка при обработке изображения: {e}")
            return False

    def _crop_and_process_image(
            self,
            image: Image.Image,
            mask: np.ndarray,
            crop_bbox: BBox
    ) -> Image.Image:
        """Вырезает и обрабатывает изображение."""
        # Кроп изображения и маски
        cropped_image = image.crop(crop_bbox.to_tuple())
        cropped_mask = mask[crop_bbox.ymin:crop_bbox.ymax,
                       crop_bbox.xmin:crop_bbox.xmax]

        # Применение маски для прозрачного фона
        cropped_with_alpha = self.image_processor.apply_mask_to_image(
            cropped_image, cropped_mask
        )

        # Создание фона
        result_image = self.image_processor.create_background_with_crop(
            crop_bbox, cropped_with_alpha, self.background_color
        )

        return result_image

    def _save_debug_image(
            self,
            image: Image.Image,
            mask: np.ndarray,
            mask_bbox: BBox,
            crop_bbox: BBox,
            output_path: Path
    ) -> None:
        """Создает и сохраняет отладочное изображение."""
        debug_image = DebugVisualizer.create_debug_image(
            image, mask, mask_bbox, crop_bbox
        )
        debug_path = output_path.parent / f"debug_{output_path.name}"
        debug_image.save(debug_path)
        self.logger.info(f"Отладочное изображение сохранено: {debug_path}")

    @staticmethod
    def _get_output_path(input_path: str) -> Path:
        """Генерирует путь для сохранения результата."""
        input_path_obj = Path(input_path)
        return input_path_obj.parent / f"{input_path_obj.stem}_face.jpg"


class Config:
    """Конфигурация приложения."""
    FIXED_PROMPT = "face"
    PAD_PERCENT = 0.0
    BACKGROUND_COLOR = (220, 220, 220)
    JPEG_QUALITY = 100


def setup_paths() -> Tuple[Path, Path]:
    """Настраивает пути к скрипту и моделям."""
    script_dir = Path(__file__).parent.resolve()
    checkpoint_path = script_dir.parent / "models" / "sam3.pt"
    return script_dir, checkpoint_path


def validate_paths(image_path: str, checkpoint_path: Path) -> bool:
    """Проверяет существование необходимых файлов."""
    if not Path(image_path).exists():
        logging.error(f"Файл изображения не найден: {image_path}")
        return False

    if not checkpoint_path.exists():
        logging.error(f"Файл весов модели не найден: {checkpoint_path}")
        return False

    return True


def main():
    """Основная функция приложения."""
    parser = argparse.ArgumentParser(
        description="Скрипт для вырезания головы плотным квадратным кропом на светло-серый фон."
    )
    parser.add_argument("image_path", type=str, help="Путь к исходному изображению.")
    parser.add_argument("--debug", action="store_true", help="Сохранить отладочное изображение.")
    parser.add_argument("--verbose", action="store_true", help="Включить подробный вывод.")

    args = parser.parse_args()

    # Настройка логирования
    Logger.setup(args.verbose)
    logger = logging.getLogger(__name__)

    # Настройка путей
    _, checkpoint_path = setup_paths()

    # Валидация путей
    if not validate_paths(args.image_path, checkpoint_path):
        sys.exit(1)

    # Определение устройства
    device_str, device_name = HeadCropSAM3.get_device()
    logger.info(f"Используемое устройство: {device_name}")

    try:
        # Создание процессора
        processor = HeadCropProcessor(
            checkpoint_path=str(checkpoint_path),
            device=device_str,
            pad_percent=Config.PAD_PERCENT,
            prompt=Config.FIXED_PROMPT,
            background_color=Config.BACKGROUND_COLOR
        )

        # Обработка изображения
        success = processor.process(args.image_path, args.debug)

        if success:
            logger.info("Готово!")
        else:
            logger.error("Обработка завершилась с ошибкой")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()