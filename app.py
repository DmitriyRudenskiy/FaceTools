from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import os


class ImageLoader(ABC):
    """Абстракция для загрузки изображений"""

    @abstractmethod
    def load(self, path: str) -> object:
        """Загружает изображение из файла"""
        pass


class ImageSaver(ABC):
    """Абстракция для сохранения изображений"""

    @abstractmethod
    def save(self, image: object, path: str) -> None:
        """Сохраняет изображение в файл"""
        pass


class FaceDetectionModel(ABC):
    """Абстракция для модели детекции лиц"""

    @abstractmethod
    def detect(self, image: object) -> List[List[float]]:
        """Детектирует лица и возвращает bounding boxes в формате [x1, y1, x2, y2]"""
        pass


class BoundingBoxProcessor(ABC):
    """Абстракция для обработки bounding box'ов"""

    @abstractmethod
    def merge_overlapping(self, boxes: List[List[float]], iou_threshold: float = 0.5) -> List[List[float]]:
        """Объединяет пересекающиеся bounding box'ы"""
        pass

    @abstractmethod
    def calculate_square_crop(self, bbox: List[float], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Рассчитывает координаты квадратной обрезки с заданным процентом отступа"""
        pass


class FileSystem(ABC):
    """Абстракция для работы с файловой системой"""

    @abstractmethod
    def get_image_files(self, path: str) -> List[str]:
        """Возвращает список путей к изображениям в указанном пути"""
        pass

    @abstractmethod
    def create_directory(self, path: str) -> None:
        """Создает директорию, если она не существует"""
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Проверяет существование пути"""
        pass

    @abstractmethod
    def is_directory(self, path: str) -> bool:
        """Проверяет, является ли путь директорией"""
        pass

    @abstractmethod
    def get_directory(self, path: str) -> str:
        """Возвращает директорию, содержащую файл"""
        pass

    @abstractmethod
    def get_basename(self, path: str) -> str:
        """Возвращает базовое имя файла без расширения"""
        pass


# Реализации для работы с изображениями
class PILImageLoader(ImageLoader):
    def load(self, path: str) -> object:
        from PIL import Image
        return Image.open(path)


class PILImageSaver(ImageSaver):
    def save(self, image: object, path: str) -> None:
        image.save(path)


# Реализация для детекции лиц
class YOLOFaceDetector(FaceDetectionModel):
    def __init__(self, model_path: str):
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def detect(self, image: object) -> List[List[float]]:
        from supervision import Detections
        output = self.model(image)
        results = Detections.from_ultralytics(output[0])
        return results.xyxy.tolist()  # Упрощенное преобразование


# Реализация для обработки bounding box'ов
class DefaultBoundingBoxProcessor(BoundingBoxProcessor):
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x1_inter >= x2_inter or y1_inter >= y2_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        return inter_area / (area1 + area2 - inter_area)

    def merge_overlapping(self, boxes: List[List[float]], iou_threshold: float = 0.5) -> List[List[float]]:
        if len(boxes) <= 1:
            return boxes

        boxes_with_areas = [(box, (box[2] - box[0]) * (box[3] - box[1])) for box in boxes]
        boxes_with_areas.sort(key=lambda x: x[1], reverse=True)
        sorted_boxes = [box for box, area in boxes_with_areas]

        merged_boxes = []
        used = [False] * len(sorted_boxes)

        for i, box1 in enumerate(sorted_boxes):
            if used[i]:
                continue

            to_merge = [box1]
            used[i] = True

            for j, box2 in enumerate(sorted_boxes[i + 1:], i + 1):
                if used[j]:
                    continue

                iou = self.calculate_iou(box1, box2)
                if iou > iou_threshold:
                    to_merge.append(box2)
                    used[j] = True

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

    def calculate_square_crop(self, bbox: List[float], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Рассчитывает координаты квадратной области для обрезки.
        Добавляет по 25% от размера квадрата с каждой стороны.
        Результат гарантированно квадратный.
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

        # Вычисляем отступы: 25% для всех сторон
        padding = int(0.25 * square_size)

        # Итоговый размер квадрата
        final_size = square_size + 2 * padding

        # Центр bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Вычисляем границы квадрата
        half_size = final_size // 2
        new_x1 = max(0, center_x - half_size)
        new_y1 = max(0, center_y - half_size)
        new_x2 = min(img_width, center_x + half_size)
        new_y2 = min(img_height, center_y + half_size)

        # Корректируем размер, если вышли за границы
        actual_width = new_x2 - new_x1
        actual_height = new_y2 - new_y1
        actual_size = min(actual_width, actual_height)

        # Корректируем размеры под квадрат
        if actual_size < final_size:
            new_x1 = max(0, center_x - actual_size // 2)
            new_x2 = new_x1 + actual_size
            if new_x2 > img_width:
                new_x2 = img_width
                new_x1 = img_width - actual_size

            new_y1 = max(0, center_y - actual_size // 2)
            new_y2 = new_y1 + actual_size
            if new_y2 > img_height:
                new_y2 = img_height
                new_y1 = img_height - actual_size

        return (int(new_x1), int(new_y1), int(new_x2), int(new_y2))


# Реализация для файловой системы
class OSFileSystem(FileSystem):
    def get_image_files(self, path: str) -> List[str]:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}

        if not self.exists(path):
            return []

        if not self.is_directory(path):
            _, ext = os.path.splitext(path.lower())
            return [path] if ext in image_extensions else []

        images = []
        for entry in os.scandir(path):
            if entry.is_file():
                _, ext = os.path.splitext(entry.name.lower())
                if ext in image_extensions:
                    images.append(entry.path)
        return images

    def create_directory(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def is_directory(self, path: str) -> bool:
        return os.path.isdir(path)

    def get_directory(self, path: str) -> str:
        return os.path.dirname(path)

    def get_basename(self, path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]


class FaceProcessingApplication:
    """Основной класс приложения, координирующий работу всех компонентов"""

    def __init__(self,
                 file_system: FileSystem,
                 face_detector: FaceDetectionModel,
                 image_loader: ImageLoader,
                 image_saver: ImageSaver,
                 bbox_processor: BoundingBoxProcessor):
        self.file_system = file_system
        self.face_detector = face_detector
        self.image_loader = image_loader
        self.image_saver = image_saver
        self.bbox_processor = bbox_processor

    def process_images(self, input_path: str, output_dir: Optional[str] = None) -> bool:
        """Обрабатывает изображения по указанному пути"""
        if not self.file_system.exists(input_path):
            print(f"Ошибка: Путь '{input_path}' не существует")
            return False

        image_files = self.file_system.get_image_files(input_path)
        if not image_files:
            print("Ошибка: Не найдено изображений для обработки")
            return False

        # Определяем директорию для сохранения
        if output_dir is None:
            if self.file_system.is_directory(input_path):
                output_dir = input_path
            else:
                output_dir = self.file_system.get_directory(input_path)

        # Создаем директорию для сохранения
        self.file_system.create_directory(output_dir)
        print(f"Директория для сохранения результатов: {output_dir}")

        # Обрабатываем каждое изображение
        for img_path in image_files:
            print(f"\nОбработка файла: {img_path}")
            try:
                # Загрузка изображения
                original_image = self.image_loader.load(img_path)
                image_size = original_image.size

                # Детекция лиц
                boxes = self.face_detector.detect(original_image)

                # Обработка bounding box'ов
                merged_boxes = self.bbox_processor.merge_overlapping(boxes)
                print(f"Найдено {len(boxes)} лиц, объединено в {len(merged_boxes)} областей")

                # Обрезка и сохранение лиц
                image_name = self.file_system.get_basename(img_path)
                for i, bbox in enumerate(merged_boxes):
                    # Рассчитываем координаты квадратной области
                    crop_coords = self.bbox_processor.calculate_square_crop(bbox, image_size)

                    # Проверка валидности координат
                    if crop_coords[2] <= crop_coords[0] or crop_coords[3] <= crop_coords[1]:
                        print(f"Пропуск невалидной области: {crop_coords}")
                        continue

                    # Обрезаем квадратную область
                    face_image = original_image.crop(crop_coords)

                    # Сохраняем обрезанное лицо
                    face_filename = f"{image_name}_face_{i + 1}.jpg"
                    face_filepath = os.path.join(output_dir, face_filename)
                    self.image_saver.save(face_image, face_filepath)
                    print(f"Сохранено лицо {i + 1}: {face_filepath}")

            except Exception as e:
                print(f"Ошибка обработки файла {img_path}: {e}")

        print("\nОбработка завершена!")
        return True


class ApplicationFactory:
    """Фабрика для создания экземпляров приложения с правильными зависимостями"""

    @staticmethod
    def create_default(model_path: str = None) -> FaceProcessingApplication:
        """Создает экземпляр приложения с дефолтными зависимостями"""
        # Определяем путь к модели, если не указан
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "model.pt")

        # Создаем зависимости
        file_system = OSFileSystem()
        face_detector = YOLOFaceDetector(model_path)
        image_loader = PILImageLoader()
        image_saver = PILImageSaver()
        bbox_processor = DefaultBoundingBoxProcessor()

        return FaceProcessingApplication(
            file_system,
            face_detector,
            image_loader,
            image_saver,
            bbox_processor
        )


def main():
    import sys

    # Проверяем количество аргументов
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Использование: python app.py <путь_к_файлу_или_директории> [директория_для_сохранения]")
        print("Пример: python app.py /Users/user/Downloads/image.jpg")
        print("Пример: python app.py /Users/user/Downloads/images/")
        print("Пример: python app.py /Users/user/Downloads/image.jpg /Users/user/Results/")
        print("Пример: python app.py /Users/user/Downloads/images/ /Users/user/Results/")
        return 1

    # Получаем пути из аргументов командной строки
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        # Создаем приложение
        app = ApplicationFactory.create_default()
        print(f"Начинаем обработку: {image_path}")

        # Запускаем обработку
        success = app.process_images(image_path, output_dir)
        return 0 if success else 1
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())