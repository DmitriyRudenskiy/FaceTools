import cv2
import numpy as np
from collections import defaultdict
import os
from abc import ABC, abstractmethod
import time

class FaceDetector(ABC):
    """Абстрактный базовый класс для детекторов лиц"""
    @abstractmethod
    def detect(self, image):
        """
        Обнаруживает лица на изображении.
        Args:
            image: Путь к изображению или numpy array изображения
        Returns:
            Список bounding boxes в формате [x1, y1, x2, y2]
        """
        pass

    def _convert_to_standard_bbox(self, bbox, format_type):
        """
        Конвертирует bounding box в стандартный формат [x1, y1, x2, y2]
        format_type может быть:
        - 'mtcnn': [x, y, width, height]
        - 'yolo11': [x1, y1, x2, y2] (уже в нужном формате)
        - 'deepface': {'x': x, 'y': y, 'w': w, 'h': h}
        - 'face_recognition': (top, right, bottom, left)
        """
        try:
            if format_type == 'mtcnn':
                x, y, w, h = bbox
                return [max(0, x), max(0, y), max(0, x + w), max(0, y + h)]
            elif format_type == 'yolo11':
                # Уже в формате [x1, y1, x2, y2]
                return bbox
            elif format_type == 'deepface':
                x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
                return [max(0, x), max(0, y), max(0, x + w), max(0, y + h)]
            elif format_type == 'face_recognition':
                top, right, bottom, left = bbox
                return [max(0, left), max(0, top), max(0, right), max(0, bottom)]
            else:
                raise ValueError(f"Unknown format type: {format_type}")
        except Exception as e:
            print(f"Error converting bbox {bbox} with format {format_type}: {e}")
            return None


class MTCNNDetector(FaceDetector):
    """Детектор лиц на основе MTCNN"""
    def __init__(self):
        self.initialized = False
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
            self.initialized = True
        except ImportError:
            print("MTCNN не установлен. Установите с помощью: pip install mtcnn")

    def detect(self, image):
        if not self.initialized:
            return []
        # Загрузка изображения если передан путь
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image
        try:
            faces = self.detector.detect_faces(img)
            result = []
            for face in faces:
                bbox = self._convert_to_standard_bbox(face['box'], 'mtcnn')
                if bbox:
                    result.append(bbox)
            return result
        except Exception as e:
            print(f"Ошибка при работе с MTCNN: {e}")
            return []


class YOLO11Detector(FaceDetector):
    """Детектор лиц на основе YOLO11 (pose-модель)"""
    def __init__(self, model_size='n'):
        """
        Инициализация YOLO11 детектора.
        Args:
            model_size: Размер модели ('n', 's', 'm', 'l', 'x')
        """
        self.initialized = False
        self.model_size = model_size
        try:
            from ultralytics import YOLO
            # Загружаем pose-модель, так как она лучше обнаруживает человека и его ключевые точки
            model_name = f"yolo11{model_size}-pose.pt"
            print(f"Загрузка YOLO11 {model_size}-pose модели. Это может занять некоторое время...")
            start_time = time.time()
            self.model = YOLO(model_name)
            load_time = time.time() - start_time
            print(f"YOLO11 {model_size}-pose модель загружена за {load_time:.2f} секунд")
            self.initialized = True
        except ImportError:
            print("Ultralytics не установлен. Установите с помощью: pip install ultralytics")
        except Exception as e:
            print(f"Ошибка при загрузке YOLO11 модели: {e}")

    def detect(self, image):
        if not self.initialized:
            return []
        try:
            # Обнаружение с использованием YOLO11
            results = self.model(image, verbose=False)
            result_boxes = []
            for result in results:
                # Получаем bounding boxes
                boxes = result.boxes
                # Обрабатываем каждую обнаруженную область
                for i in range(len(boxes)):
                    # Получаем координаты bounding box
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    # Проверяем, что это человек (класс 0 в COCO)
                    # В pose-модели YOLO11 обнаруживает только людей
                    if int(boxes.cls[i]) == 0:
                        bbox = self._convert_to_standard_bbox([x1, y1, x2, y2], 'yolo11')
                        if bbox:
                            result_boxes.append(bbox)
            return result_boxes
        except Exception as e:
            print(f"Ошибка при работе с YOLO11: {e}")
            return []


class DeepFaceDetector(FaceDetector):
    """Детектор лиц на основе DeepFace"""
    def __init__(self):
        self.initialized = False
        try:
            from deepface import DeepFace
            self.detector = DeepFace
            self.initialized = True
        except ImportError:
            print("DeepFace не установлен. Установите с помощью: pip install deepface")

    def detect(self, image):
        if not self.initialized:
            return []
        try:
            # Если передан путь к файлу
            if isinstance(image, str) and os.path.exists(image):
                faces = self.detector.extract_faces(img_path=image,
                                                 detector_backend='opencv',
                                                 enforce_detection=False)
            else:
                # Создаем временный файл для обработки
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                    temp_path = f.name
                    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                faces = self.detector.extract_faces(img_path=temp_path,
                                                 detector_backend='opencv',
                                                 enforce_detection=False)
                os.unlink(temp_path)
            result = []
            for face in faces:
                bbox = self._convert_to_standard_bbox(face['region'], 'deepface')
                if bbox:
                    result.append(bbox)
            return result
        except Exception as e:
            print(f"Ошибка при работе с DeepFace: {e}")
            return []


class FaceRecognitionDetector(FaceDetector):
    """Детектор лиц на основе face_recognition"""
    def __init__(self):
        self.initialized = False
        try:
            import face_recognition
            self.detector = face_recognition
            self.initialized = True
        except ImportError:
            print("face_recognition не установлен. Установите с помощью: pip install face_recognition")

    def detect(self, image):
        if not self.initialized:
            return []
        try:
            # Если передан путь к файлу
            if isinstance(image, str) and os.path.exists(image):
                img = cv2.imread(image)
                if img is None:
                    raise ValueError(f"Could not load image from {image}")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = image
            face_locations = self.detector.face_locations(img_rgb)
            result = []
            for location in face_locations:
                bbox = self._convert_to_standard_bbox(location, 'face_recognition')
                if bbox:
                    result.append(bbox)
            return result
        except Exception as e:
            print(f"Ошибка при работе с Face_recognition: {e}")
            return []


class EnsembleFaceDetector:
    """Ансамблевый детектор лиц, объединяющий результаты нескольких моделей"""
    def __init__(self, iou_threshold=0.5, min_votes=3, use_voting=True, save_images=False):
        """
        Инициализация ансамблевого детектора.
        Args:
            iou_threshold: порог IoU для определения совпадения лиц
            min_votes: минимальное количество голосов для включения лица в результат
            use_voting: флаг использования голосования моделей
            save_images: флаг сохранения изображений с bounding box'ами
        """
        self.iou_threshold = iou_threshold
        self.min_votes = min_votes
        self.use_voting = use_voting
        self.save_images = save_images
        # Инициализация всех детекторов
        self.detectors = {
            'mtcnn': MTCNNDetector(),
            'yolo11': YOLO11Detector(model_size='x'),
            'face_recognition': FaceRecognitionDetector()
        }
        # Фильтрация неработающих детекторов
        self.active_detectors = {name: detector for name, detector in self.detectors.items()
                                if detector.initialized}
        if not self.active_detectors:
            raise RuntimeError("Ни один из детекторов не был успешно инициализирован. Проверьте установку зависимостей.")
        print(f"Инициализировано {len(self.active_detectors)} активных детекторов: {', '.join(self.active_detectors.keys())}")
        if self.use_voting:
            print(f"Будет использоваться голосование: минимум {self.min_votes} из {len(self.active_detectors)} моделей")
        else:
            print("Будет использоваться объединение результатов всех моделей (без голосования)")

    @staticmethod
    def iou(box1, box2, epsilon=1e-5):
        """
        Вычисляет Intersection over Union (IoU) между двумя bounding box'ами.
        Box формата [x1, y1, x2, y2]
        """
        # Определяем координаты пересечения
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        # Площадь пересечения
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        # Площади каждого из прямоугольников
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        # IoU
        union = area1 + area2 - intersection
        return intersection / (union + epsilon)

    def detect(self, image):
        """
        Обнаруживает лица с использованием всех доступных моделей и применяет голосование.
        Args:
            image: Путь к изображению или numpy array изображения
        Returns:
            Список кортежей (bbox, cropped_face), где:
            - bbox: координаты лица в формате [x1, y1, x2, y2]
            - cropped_face: изображение лица в виде numpy array
        """
        # Список для хранения всех обнаруженных лиц
        all_detections = []
        original_image_path = None

        # Загрузка изображения если передан путь
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            original_image_path = image
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from {image}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Обнаружение лиц каждой моделью
        for detector_name, detector in self.active_detectors.items():
            try:
                start_time = time.time()
                bboxes = detector.detect(image)
                detection_time = time.time() - start_time
                print(f"{detector_name}: обнаружено {len(bboxes)} лиц (время: {detection_time:.2f} сек)")
                for bbox in bboxes:
                    all_detections.append((detector_name, bbox))
            except Exception as e:
                print(f"Ошибка при обнаружении лиц с {detector_name}: {e}")

        # Если ни одна модель не обнаружила лица, возвращаем пустой список
        if not all_detections:
            print("Ни одна модель не обнаружила лица")
            return []

        # Группируем обнаружения по источнику
        detections_by_source = defaultdict(list)
        for source, bbox in all_detections:
            detections_by_source[source].append(bbox)

        # Создаем список всех уникальных обнаружений
        all_unique_detections = []
        for _, bboxes in detections_by_source.items():
            all_unique_detections.extend(bboxes)

        # Удаляем дубликаты с помощью нечеткого сравнения
        unique_faces = []
        for i, bbox1 in enumerate(all_unique_detections):
            is_unique = True
            for j, bbox2 in enumerate(unique_faces):
                if self.iou(bbox1, bbox2) > self.iou_threshold:
                    is_unique = False
                    break
            if is_unique:
                unique_faces.append(bbox1)

        # Для каждого уникального обнаружения проверяем, сколько моделей его обнаружили
        final_results = []
        rejected_faces = []
        print("\n" + "=" * 50)
        print("Анализ обнаруженных лиц:")
        print("=" * 50)
        for face in unique_faces:
            votes = 0
            sources = []
            for source, bboxes in detections_by_source.items():
                for bbox in bboxes:
                    if self.iou(face, bbox) > self.iou_threshold:
                        votes += 1
                        sources.append(source)
                        break  # Одно совпадение от источника достаточно
            x1, y1, x2, y2 = map(int, face)
            # Проверяем, достаточно ли голосов
            if votes >= self.min_votes or not self.use_voting:
                # Извлекаем область лица
                cropped_face = img_rgb[y1:y2, x1:x2]
                # Добавляем в результат
                final_results.append((face, cropped_face))
                print(f"✅ Лицо [{x1},{y1},{x2},{y2}] ПОДТВЕРЖДЕНО {votes} моделями: {', '.join(sources)}")
            else:
                rejected_faces.append((face, votes, sources))
                print(f"❌ Лицо [{x1},{y1},{x2},{y2}] ПРОИГНОРИРОВАНО ({votes} голос/а из {self.min_votes} требуемых): {', '.join(sources)}")

        # Вывод статистики
        print("\n" + "=" * 50)
        print("Статистика голосования:")
        print(f"Всего уникальных обнаружений: {len(unique_faces)}")
        print(f"Принято: {len(final_results)} лиц")
        print(f"Проигнорировано: {len(rejected_faces)} лиц")
        # Детальная статистика по проигнорированным лицам
        if rejected_faces:
            print("\nДетали проигнорированных лиц:")
            for i, (face, votes, sources) in enumerate(rejected_faces):
                x1, y1, x2, y2 = map(int, face)
                print(f"  {i + 1}. [{x1},{y1},{x2},{y2}] - {votes} голос/а: {', '.join(sources)}")
        print(f"\nИтог: обнаружено {len(final_results)} лиц после голосования (минимум {self.min_votes} моделей подтвердили)")
        return final_results

    def visualize_results(self, image, results, output_path=None):
        """
        Визуализирует результаты обнаружения лиц.
        Args:
            image: Исходное изображение
            results: Результаты из метода detect
            output_path: Путь для сохранения изображения (опционально)
        Returns:
            Изображение с нарисованными bounding box'ами
        """
        import matplotlib.pyplot as plt
        # Загружаем изображение если передан путь
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()

        plt.figure(figsize=(12, 8))
        for i, (bbox, _) in enumerate(results):
            x1, y1, x2, y2 = map(int, bbox)
            # Рисуем прямоугольник
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Добавляем номер лица
            cv2.putText(img, f"#{i + 1}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        plt.imshow(img)
        plt.axis('off')
        if self.use_voting:
            plt.title(f"Обнаружено {len(results)} лиц (подтверждено минимум {self.min_votes} моделями)")
        else:
            plt.title(f"Обнаружено {len(results)} лиц (без голосования)")
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Результат сохранен в {output_path}")
        # Убираем автоматическое отображение графиков
        # plt.show()
        return img

    def save_bounding_box_image(self, image, results, output_dir=None):
        """
        Сохраняет изображение с нарисованными bounding box'ами.
        Args:
            image: Исходное изображение
            results: Результаты из метода detect
            output_dir: Папка для сохранения результата (по умолчанию - та же директория, что и у исходного изображения)
        """
        import matplotlib.pyplot as plt
        # Загружаем изображение если передан путь
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()

        for i, (bbox, _) in enumerate(results):
            x1, y1, x2, y2 = map(int, bbox)
            # Рисуем прямоугольник
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Добавляем номер лица
            cv2.putText(img, f"#{i + 1}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Определяем путь для сохранения
        if isinstance(image, str):
            base_dir = os.path.dirname(image)
            filename = os.path.splitext(os.path.basename(image))[0]
            output_dir = output_dir or base_dir
            output_path = os.path.join(output_dir, f"{filename}_bbox.jpg")
        else:
            output_path = "output_bbox.jpg"

        # Сохраняем изображение
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Изображение с bounding box'ами сохранено в: {output_path}")


# Пример использования
if __name__ == "__main__":
    # Инициализация ансамблевого детектора
    detector = EnsembleFaceDetector(
        iou_threshold=0.5,
        min_votes=3,
        use_voting=False,  # Отключаем голосование
        save_images=True    # Включаем сохранение изображений с bounding box'ами
    )

    # Путь к изображению
    image_path = "/Users/user/Downloads/73909664-8c012a7e3dd701275d775b6c455c59fe8d641f609d7fb2a902be865964290bb2.png"  # Замените на путь к вашему изображению

    try:
        # Обнаружение лиц
        print(f"\n{'=' * 50}")
        print(f"Начало обнаружения лиц на изображении: {image_path}")
        print(f"{'=' * 50}")
        start_time = time.time()
        results = detector.detect(image_path)
        total_time = time.time() - start_time
        print(f"\n{'=' * 50}")
        print(f"Обработка завершена за {total_time:.2f} секунд")
        print(f"{'=' * 50}")

        # Визуализация результатов
        if results:
            output_path = image_path.rsplit('.', 1)[0] + "_result.jpg"
            detector.visualize_results(image_path, results, output_path)

            # Сохраняем изображение с bounding box'ами
            if detector.save_images:
                detector.save_bounding_box_image(image_path, results)

            # Дополнительно: показать изображения обнаруженных лиц
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 5))
            for i, (_, face_img) in enumerate(results):
                plt.subplot(1, len(results), i + 1)
                plt.imshow(face_img)
                plt.title(f"Лицо #{i + 1}")
                plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            print("Лица не обнаружены после голосования")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        import traceback
        traceback.print_exc()