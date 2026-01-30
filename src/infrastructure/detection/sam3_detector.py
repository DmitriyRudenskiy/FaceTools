"""
Модуль детектора лиц на основе SAM (Segment Anything Model).
Реализует интерфейс FaceDetector для обнаружения лиц с использованием SAM.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple
import numpy as np
from PIL import Image

from ultralytics.models.sam import SAM3SemanticPredictor
from src.core.interfaces import FaceDetector
from src.core.exceptions import FaceDetectionError


class SAMFaceDetector(FaceDetector):
    """Детекция лиц с использованием SAM (Segment Anything Model)"""

    def __init__(
            self,
            model_path: Optional[str] = None,
            device: Optional[str] = None,
            prompt_text: str = "head",
            confidence_threshold: float = 0.5
    ) -> None:
        """
        Инициализация модели SAM.

        Args:
            model_path: Путь к файлу модели SAM (.pt)
            device: Устройство для вычислений ('cpu', 'cuda', 'mps')
            prompt_text: Текстовый промпт для детекции лиц
            confidence_threshold: Порог уверенности для детекции
        """
        self.prompt_text = prompt_text
        self.confidence_threshold = confidence_threshold

        # Установка пути к модели по умолчанию
        if model_path is None:
            project_root = Path(__file__).resolve().parents[3]
            model_path = str(project_root / "models" / "sam3.pt")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        # Определение устройства
        if device is None:
            device = self._detect_device()

        self.device = device
        self.model = self._load_model()

    def _detect_device(self) -> str:
        """Автоматическое определение доступного устройства."""
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> SAM3SemanticPredictor:
        """Загрузка модели SAM с настройками."""
        try:
            overrides = {
                'conf': self.confidence_threshold,
                'task': 'segment',
                'mode': 'predict',
                'model': str(self.model_path),
                'device': self.device,
                'imgsz': 1024,  # Размер изображения для SAM
                'save': False,
                'half': False,
                'verbose': False
            }

            return SAM3SemanticPredictor(overrides=overrides)
        except Exception as e:
            raise FaceDetectionError(f"Ошибка загрузки модели SAM: {str(e)}") from e

    def detect(self, image: Any) -> List[List[float]]:
        """
        Обнаружение лиц на изображении с использованием SAM.

        Args:
            image: Входное изображение (PIL.Image, np.ndarray или путь к файлу)

        Returns:
            Список bounding boxes в формате [x1, y1, x2, y2]
        """
        try:
            # Преобразование входных данных в PIL Image если нужно
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Получение размеров изображения
            img_width, img_height = image.size

            # Установка изображения для модели SAM
            self.model.set_image(image)

            # Выполнение сегментации с текстовым промптом
            results = self.model(text=[self.prompt_text])

            if not results or results[0].masks is None:
                return []

            # Извлечение масок и преобразование в bounding boxes
            masks = results[0].masks.data.cpu().numpy()
            boxes = []

            for i in range(masks.shape[0]):
                mask = masks[i]

                # Получение bounding box из маски
                bbox = self._mask_to_bbox(mask)
                if bbox:
                    # Масштабирование координат к размеру изображения
                    scale_x = img_width / mask.shape[1]
                    scale_y = img_height / mask.shape[0]

                    x1, y1, x2, y2 = bbox
                    boxes.append([
                        x1 * scale_x,
                        y1 * scale_y,
                        x2 * scale_x,
                        y2 * scale_y
                    ])

            # Фильтрация пересекающихся боксов (NMS)
            filtered_boxes = self._non_max_suppression(boxes)

            return filtered_boxes

        except Exception as e:
            print(f"Ошибка детекции лиц SAM: {str(e)}")
            return []

    def _mask_to_bbox(self, mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
        Преобразование бинарной маски в bounding box.

        Args:
            mask: Бинарная маска (2D numpy array)

        Returns:
            Координаты bbox в формате (x1, y1, x2, y2) или None
        """
        # Бинаризация маски
        binary_mask = mask > 0.5

        if not np.any(binary_mask):
            return None

        # Находим ненулевые пиксели
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)

        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]

        if len(y_indices) == 0 or len(x_indices) == 0:
            return None

        y1 = float(y_indices[0])
        y2 = float(y_indices[-1])
        x1 = float(x_indices[0])
        x2 = float(x_indices[-1])

        return (x1, y1, x2, y2)

    def _non_max_suppression(
            self,
            boxes: List[List[float]],
            iou_threshold: float = 0.5
    ) -> List[List[float]]:
        """
        Non-Maximum Suppression для фильтрации пересекающихся bounding boxes.

        Args:
            boxes: Список bounding boxes
            iou_threshold: Порог IoU для удаления пересекающихся боксов

        Returns:
            Отфильтрованный список bounding boxes
        """
        if not boxes:
            return []

        # Конвертируем в numpy для удобства
        boxes_array = np.array(boxes)

        # Вычисляем площади
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 2]
        y2 = boxes_array[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Сортируем по y2 координате (нижней границе)
        indices = np.argsort(y2)
        keep = []

        while len(indices) > 0:
            # Берем последний бокс (самый нижний)
            i = indices[-1]
            keep.append(i)

            if len(indices) == 1:
                break

            # Оставшиеся боксы
            remaining = indices[:-1]

            # Вычисляем IoU с оставшимися боксами
            xx1 = np.maximum(x1[i], x1[remaining])
            yy1 = np.maximum(y1[i], y1[remaining])
            xx2 = np.minimum(x2[i], x2[remaining])
            yy2 = np.minimum(y2[i], y2[remaining])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            intersection = w * h
            union = areas[i] + areas[remaining] - intersection
            iou = intersection / union

            # Удаляем боксы с высоким IoU
            indices = remaining[iou <= iou_threshold]

        return boxes_array[keep].tolist()

    def extract_embeddings(self, image: Any) -> List[np.ndarray]:
        """
        Извлекает эмбеддинги для всех обнаруженных лиц на изображении.

        Note: SAM не предоставляет эмбеддинги лиц напрямую.
        Этот метод возвращает эмбеддинги масок вместо эмбеддингов лиц.

        Args:
            image: Входное изображение для обработки

        Returns:
            Список эмбеддингов для каждого обнаруженного лица (масок)
        """
        try:
            # Преобразование входных данных
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Установка изображения для модели SAM
            self.model.set_image(image)

            # Выполнение сегментации
            results = self.model(text=[self.prompt_text])

            if not results or results[0].masks is None:
                return []

            # Извлечение масок и их использование как "эмбеддингов"
            masks = results[0].masks.data.cpu().numpy()
            embeddings = []

            for i in range(masks.shape[0]):
                mask = masks[i]
                # Используем flattened маску как эмбеддинг
                embedding = mask.flatten()
                # Нормализуем
                if embedding.std() > 0:
                    embedding = (embedding - embedding.mean()) / embedding.std()
                embeddings.append(embedding)

            return embeddings

        except Exception as e:
            raise FaceDetectionError(f"Ошибка извлечения эмбеддингов SAM: {str(e)}") from e

    def compare_faces(
            self,
            image1: Any,
            image2: Any,
            threshold: float = 0.6
    ) -> Tuple[bool, float]:
        """
        Сравнивает лица на двух изображениях.

        Note: SAM не предназначен для сравнения лиц. Этот метод использует
        сравнение масок, что не является оптимальным для верификации лиц.

        Args:
            image1: Первое изображение с лицом
            image2: Второе изображение с лицом
            threshold: Пороговое значение схожести

        Returns:
            Кортеж (bool, float): Результат сравнения и значение схожести
        """
        try:
            # Извлекаем эмбеддинги
            emb1_list = self.extract_embeddings(image1)
            emb2_list = self.extract_embeddings(image2)

            if not emb1_list or not emb2_list:
                return (False, 0.0)

            # Берем первые обнаруженные лица
            emb1 = emb1_list[0]
            emb2 = emb2_list[0]

            # Убедимся, что эмбеддинги имеют одинаковую размерность
            min_len = min(len(emb1), len(emb2))
            emb1 = emb1[:min_len]
            emb2 = emb2[:min_len]

            # Вычисляем косинусную схожесть
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(
                emb1.reshape(1, -1),
                emb2.reshape(1, -1)
            )[0][0]

            # Нормализуем в диапазон [0, 1]
            normalized_similarity = (similarity + 1) / 2

            return (normalized_similarity >= threshold, normalized_similarity)

        except Exception as e:
            raise FaceDetectionError(f"Ошибка сравнения лиц SAM: {str(e)}") from e

    def detect_with_masks(self, image: Any) -> Tuple[List[List[float]], List[np.ndarray]]:
        """
        Обнаружение лиц вместе с масками сегментации.

        Args:
            image: Входное изображение

        Returns:
            Кортеж (bounding_boxes, masks)
        """
        try:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            img_width, img_height = image.size

            # Установка изображения и сегментация
            self.model.set_image(image)
            results = self.model(text=[self.prompt_text])

            if not results or results[0].masks is None:
                return [], []

            masks = results[0].masks.data.cpu().numpy()
            boxes = []
            mask_list = []

            for i in range(masks.shape[0]):
                mask = masks[i]

                # Получение bounding box
                bbox = self._mask_to_bbox(mask)
                if bbox:
                    # Масштабирование
                    scale_x = img_width / mask.shape[1]
                    scale_y = img_height / mask.shape[0]

                    x1, y1, x2, y2 = bbox
                    boxes.append([
                        x1 * scale_x,
                        y1 * scale_y,
                        x2 * scale_x,
                        y2 * scale_y
                    ])

                    # Сохранение маски (масштабированной к размеру изображения)
                    from scipy.ndimage import zoom
                    scaled_mask = zoom(mask, (scale_y, scale_x), order=1)
                    mask_list.append(scaled_mask)

            return boxes, mask_list

        except Exception as e:
            print(f"Ошибка детекции с масками: {str(e)}")
            return [], []

    def refine_bbox_with_mask(
            self,
            image: Any,
            bbox: List[float]
    ) -> List[float]:
        """
        Уточнение bounding box с использованием маски сегментации.

        Args:
            image: Входное изображение
            bbox: Исходный bounding box [x1, y1, x2, y2]

        Returns:
            Уточненный bounding box
        """
        try:
            # Кроп изображения по bbox
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            x1, y1, x2, y2 = map(int, bbox)
            cropped = image.crop((x1, y1, x2, y2))

            # Сегментация на кропе
            self.model.set_image(cropped)
            results = self.model(text=[self.prompt_text])

            if not results or results[0].masks is None:
                return bbox

            # Получение маски и уточнение bbox
            mask = results[0].masks.data[0].cpu().numpy()
            refined_bbox = self._mask_to_bbox(mask)

            if refined_bbox:
                # Масштабирование обратно к исходным координатам
                rx1, ry1, rx2, ry2 = refined_bbox
                scale_x = (x2 - x1) / mask.shape[1]
                scale_y = (y2 - y1) / mask.shape[0]

                return [
                    x1 + rx1 * scale_x,
                    y1 + ry1 * scale_y,
                    x1 + rx2 * scale_x,
                    y1 + ry2 * scale_y
                ]

            return bbox

        except Exception as e:
            print(f"Ошибка уточнения bbox: {str(e)}")
            return bbox


class SAMBoundingBoxProcessor:
    """Процессор для обработки bounding boxes от SAM детектора."""

    def merge_overlapping(
            self,
            boxes: List[List[float]],
            iou_threshold: float = 0.5
    ) -> List[List[float]]:
        """
        Слияние пересекающихся bounding boxes.

        Args:
            boxes: Список bounding boxes
            iou_threshold: Порог IoU для слияния

        Returns:
            Список объединенных bounding boxes
        """
        if not boxes:
            return []

        boxes_array = np.array(boxes)

        # Вычисляем IoU между всеми парами боксов
        iou_matrix = self._calculate_iou_matrix(boxes_array)

        # Объединение пересекающихся боксов
        merged_boxes = []
        used = set()

        for i in range(len(boxes_array)):
            if i in used:
                continue

            # Находим все пересекающиеся боксы
            overlapping = np.where(iou_matrix[i] > iou_threshold)[0]

            if len(overlapping) > 0:
                # Объединяем все пересекающиеся боксы
                group_boxes = boxes_array[overlapping]
                x1 = group_boxes[:, 0].min()
                y1 = group_boxes[:, 1].min()
                x2 = group_boxes[:, 2].max()
                y2 = group_boxes[:, 3].max()
                merged_boxes.append([x1, y1, x2, y2])

                used.update(overlapping)
            else:
                merged_boxes.append(boxes_array[i].tolist())
                used.add(i)

        return merged_boxes

    def _calculate_iou_matrix(self, boxes: np.ndarray) -> np.ndarray:
        """Вычисление матрицы IoU между всеми парами bounding boxes."""
        n = len(boxes)
        iou_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                iou = self._calculate_iou(boxes[i], boxes[j])
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou

        return iou_matrix

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Вычисление Intersection over Union для двух bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return intersection / (area1 + area2 - intersection)

    def calculate_square_crop(
            self,
            bbox: List[float],
            image_size: Tuple[int, int],
            padding_ratio: float = 0.2
    ) -> Tuple[int, int, int, int]:
        """
        Рассчитывает квадратную обрезку на основе bounding box.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            image_size: Размеры изображения (ширина, высота)
            padding_ratio: Отступ в долях от размера bbox

        Returns:
            Координаты квадратной обрезки (x1, y1, x2, y2)
        """
        img_w, img_h = image_size
        x1, y1, x2, y2 = bbox

        # Вычисляем центр bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Вычисляем размер квадрата
        width = x2 - x1
        height = y2 - y1
        max_size = max(width, height)

        # Добавляем отступы
        square_size = int(max_size * (1 + padding_ratio))

        # Вычисляем координаты квадрата
        half_size = square_size / 2
        crop_x1 = int(center_x - half_size)
        crop_y1 = int(center_y - half_size)
        crop_x2 = int(center_x + half_size)
        crop_y2 = int(center_y + half_size)

        # Проверка границ изображения
        if crop_x1 < 0:
            crop_x2 -= crop_x1
            crop_x1 = 0

        if crop_y1 < 0:
            crop_y2 -= crop_y1
            crop_y1 = 0

        if crop_x2 > img_w:
            diff = crop_x2 - img_w
            crop_x1 -= diff
            crop_x2 = img_w

        if crop_y2 > img_h:
            diff = crop_y2 - img_h
            crop_y1 -= diff
            crop_y2 = img_h

        # Корректировка для сохранения квадрата
        final_size = min(crop_x2 - crop_x1, crop_y2 - crop_y1)
        crop_x2 = crop_x1 + final_size
        crop_y2 = crop_y1 + final_size

        return (crop_x1, crop_y1, crop_x2, crop_y2)