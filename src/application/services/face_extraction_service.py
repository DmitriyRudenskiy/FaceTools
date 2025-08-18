# src/application/services/face_clustering_service.py

import os
from typing import List
from src.core.interfaces import FaceDetector, BoundingBoxProcessor, ImageLoader, FileOrganizer, ResultSaver
from src.core.exceptions import FaceDetectionError, FileHandlingError
from src.domain.image import Image
from src.domain.face import Face, BoundingBox


class FaceExtractionService:  # <<<--- ИМЯ КЛАССА ИЗМЕНЕНО ЗДЕСЬ
    """Сервис обработки и кластеризации лиц"""

    def __init__(
            self,
            file_organizer: FileOrganizer,
            face_detector: FaceDetector,
            bbox_processor: BoundingBoxProcessor,
            image_loader: ImageLoader,
            result_saver: ResultSaver
    ):
        self.file_organizer = file_organizer
        self.face_detector = face_detector
        self.bbox_processor = bbox_processor
        self.image_loader = image_loader
        self.result_saver = result_saver
        self.processing_results = []  # Для сбора результатов перед выводом таблицы

    def process(self, input_path: str, output_file: str = "groups.json", dest_dir: str = None) -> bool:
        """Основной метод обработки - совместимый с CLI"""
        return self.process_images(input_path, dest_dir)

    def process_images(self, input_path: str, output_dir: str = None) -> bool:
        """Обрабатывает изображения и сохраняет вырезанные лица.

        Args:
            input_path: Путь к изображению или директории с изображениями
            output_dir: Директория для сохранения вырезанных лиц.
                       Если None, то создаётся поддиректория 'faces' в директории источника.
        """
        try:
            if not self.file_organizer.exists(input_path):
                raise FileHandlingError(f"Путь '{input_path}' не существует")
        except Exception as e:
            print(f"Ошибка проверки пути: {str(e)}")
            return False

        # Определяем выходную директорию
        if output_dir is None:
            # Если input_path - файл, то output_dir = директория файла + 'faces'
            if not self.file_organizer.is_directory(input_path):
                source_dir = self.file_organizer.get_directory(input_path)
            else:
                # Если input_path - директория, то output_dir = input_path + 'faces'
                source_dir = input_path
            output_dir = os.path.join(source_dir, "faces")

        try:
            self.file_organizer.create_directory(output_dir)
        except Exception as e:
            print(f"Ошибка создания директории '{output_dir}': {str(e)}")
            return False

        print(f"Вырезанные лица будут сохранены в: {output_dir}")

        # Обрабатываем каждое изображение
        success_count = 0
        total_count = 0
        image_files = self.file_organizer.get_image_files(input_path)

        if not image_files:
            print("Не найдено изображений для обработки.")
            return False

        for img_path in image_files:
            total_count += 1
            try:
                filename = os.path.basename(img_path)
                image = self.image_loader.load(img_path)
                faces = self._detect_faces(image)

                # Сохраняем каждое лицо и собираем результаты
                saved_files = []
                for i, face in enumerate(faces):
                    face_img = self._crop_face(image, face.bounding_box)
                    output_path = self._generate_output_path(img_path, i, output_dir)
                    self.result_saver.save(face_img, output_path)
                    # Сохраняем только имя файла без пути
                    saved_filename = os.path.basename(output_path)
                    saved_files.append(saved_filename)

                # Сохраняем результаты для последующего вывода таблицы
                self.processing_results.append((filename, saved_files))
                success_count += 1
                print(f"Обработано: {filename} -> {len(faces)} лиц(о)")

            except FaceDetectionError as e:
                print(f"Ошибка детекции на {filename}: {str(e)}")
                continue
            except Exception as e:
                print(f"Неожиданная ошибка при обработке {filename}: {str(e)}")
                continue

        # Выводим результаты в виде таблицы
        if self.processing_results:
            self._print_results_table()

        print(f"\nОбработка завершена! Обработано: {success_count}/{total_count} изображений")
        print(f"Вырезанные лица сохранены в: {output_dir}")
        return success_count > 0

    def _print_results_table(self):
        """Выводит результаты обработки в виде таблицы"""
        if not self.processing_results:
            return

        # Определяем ширину колонок
        max_input_len = max(len("Исходное изображение"), max(len(item[0]) for item in self.processing_results))
        max_output_len = max(len("Вырезанные лица"),
                             max(len(", ".join(item[1])) if item[1] else 0 for item in self.processing_results))

        # Выводим заголовок таблицы
        print("\n" + "=" * (max_input_len + max_output_len + 7))
        print(f"| {'Исходное изображение':<{max_input_len}} | {'Вырезанные лица':<{max_output_len}} |")
        print("|" + "-" * (max_input_len + 2) + "|" + "-" * (max_output_len + 2) + "|")

        # Выводим строки таблицы
        for filename, saved_files in self.processing_results:
            saved_files_str = ", ".join(saved_files) if saved_files else "Нет лиц"
            print(f"| {filename:<{max_input_len}} | {saved_files_str:<{max_output_len}} |")
        print("=" * (max_input_len + max_output_len + 7))

    def _detect_faces(self, image: Image) -> List[Face]:
        """Детекция лиц на изображении"""
        try:
            raw_boxes = self.face_detector.detect(image)
            merged_boxes = self.bbox_processor.merge_overlapping(raw_boxes)
            # Создаем объекты Face из bounding boxes
            faces = []
            for box in merged_boxes:
                # Передаём весь объект Image, а не только data
                faces.append(Face(
                    bounding_box=box,
                    landmarks=None,  # В текущей реализации YOLO landmarks не определяются
                    embedding=[],  # embedding будет заполняться позже, если нужно
                    image=image  # Передаём объект Image
                ))
            return faces
        except Exception as e:
            raise FaceDetectionError(f"Ошибка детекции лиц: {str(e)}") from e

    def _crop_face(self, image: Image, bbox: BoundingBox) -> any:
        """Обрезка лица по bounding box"""
        try:
            crop_coords = self.bbox_processor.calculate_square_crop(
                bbox,
                (image.info.size[0], image.info.size[1])
            )
            return image.data.crop(crop_coords)
        except Exception as e:
            raise FaceDetectionError(f"Ошибка обрезки лица: {str(e)}") from e

    def _generate_output_path(self, source_path: str, face_index: int, output_dir: str) -> str:
        """Генерация пути для сохранения лица с именем вида <original_name>_face_<N>.jpg"""
        base_name = self.file_organizer.get_basename(source_path)
        return os.path.join(output_dir, f"{base_name}_face_{face_index + 1}.jpg")
