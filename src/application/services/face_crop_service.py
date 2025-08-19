import os
from typing import List

from src.core.exceptions import FaceDetectionError, FileHandlingError
from src.core.interfaces import (BoundingBoxProcessor, FaceDetector,
                                 FileOrganizer, ImageLoader)
from src.domain.face import BoundingBox, Face
from src.domain.image_model import Image  # Обратите внимание на имя модуля


class FaceCropService:
    def __init__(
        self,
        file_organizer: FileOrganizer,
        face_detector: FaceDetector,
        bbox_processor: BoundingBoxProcessor,
        image_loader: ImageLoader,
    ):
        self.file_organizer = file_organizer
        self.face_detector = face_detector
        self.bbox_processor = bbox_processor
        self.image_loader = image_loader
        self.processing_results = []  # Для сбора результатов перед выводом таблицы

    def process_images(self, input_path: str, output_dir: str = None) -> bool:
        """Обрабатывает изображения и сохраняет вырезанные лица"""
        try:
            if not self.file_organizer.exists(input_path):
                raise FileHandlingError(f"Путь '{input_path}' не существует")
        except Exception as e:
            print(f"Ошибка проверки пути: {str(e)}")
            return False

        # Определяем выходную директорию
        if output_dir is None:
            output_dir = (
                input_path
                if self.file_organizer.is_directory(input_path)
                else self.file_organizer.get_directory(input_path)
            )

        try:
            self.file_organizer.create_directory(output_dir)
        except Exception as e:
            print(f"Ошибка создания директории: {str(e)}")
            return False

        # Обрабатываем каждое изображение
        success_count = 0
        total_count = 0

        for img_path in self.file_organizer.get_image_files(input_path):
            total_count += 1
            try:
                # Извлекаем только имя файла без пути
                filename = os.path.basename(img_path)
                image = self.image_loader.load(img_path)
                faces = self._detect_faces(image)

                # Сохраняем каждое лицо и собираем результаты
                saved_files = []
                for i, face in enumerate(faces):
                    face_img = self._crop_face(image, face.bounding_box)
                    output_path = self._generate_output_path(img_path, i, output_dir)
                    self.file_organizer.save(face_img, output_path)

                    # Сохраняем только имя файла без пути
                    saved_filename = os.path.basename(output_path)
                    saved_files.append(saved_filename)

                # Сохраняем результаты для последующего вывода таблицы
                self.processing_results.append((filename, saved_files))
                success_count += 1
            except FaceDetectionError:
                continue
            except Exception:
                continue

        # Выводим результаты в виде таблицы
        self._print_results_table()

        print(
            f"\nОбработка завершена! Обработано: {success_count}/{total_count} изображений"
        )
        return success_count > 0

    def _print_results_table(self):
        """Выводит результаты обработки в виде таблицы"""
        if not self.processing_results:
            return

        # Определяем ширину колонок
        max_input_len = max(
            len("Обработка"), max(len(item[0]) for item in self.processing_results)
        )
        max_output_len = max(
            len("Сохранено"),
            max(len(", ".join(item[1])) for item in self.processing_results),
        )

        # Выводим заголовок таблицы
        print("\n" + "=" * (max_input_len + max_output_len + 7))
        print(f"| {'Обработка':<{max_input_len}} | {'Сохранено':<{max_output_len}} |")
        print("|" + "-" * (max_input_len + 2) + "|" + "-" * (max_output_len + 2) + "|")

        # Выводим строки таблицы
        for filename, saved_files in self.processing_results:
            saved_files_str = ", ".join(saved_files) if saved_files else ""
            print(
                f"| {filename:<{max_input_len}} | {saved_files_str:<{max_output_len}} |"
            )

        print("=" * (max_input_len + max_output_len + 7))

    def _detect_faces(self, image: Image) -> List[Face]:
        """Детекция лиц на изображении"""
        try:
            raw_boxes = self.face_detector.detect(image)
            merged_boxes = self.bbox_processor.merge_overlapping(raw_boxes)

            # Создаем объекты Face из bounding boxes
            faces = []
            for box in merged_boxes:
                faces.append(
                    Face(bounding_box=box, landmarks=None, embedding=[], image=image)
                )
            return faces
        except Exception as e:
            raise FaceDetectionError(f"Ошибка детекции лиц: {str(e)}") from e

    def _crop_face(self, image: Image, bbox: BoundingBox) -> any:
        """Обрезка лица по bounding box"""
        try:
            crop_coords = self.bbox_processor.calculate_square_crop(
                bbox, (image.info.size[0], image.info.size[1])
            )
            return image.data.crop(crop_coords)
        except Exception as e:
            raise FaceDetectionError(f"Ошибка обрезки лица: {str(e)}") from e

    def _generate_output_path(
        self, source_path: str, face_index: int, output_dir: str
    ) -> str:
        """Генерация пути для сохранения лица с именем вида original_face_N.jpg"""
        base_name = self.file_organizer.get_basename(source_path)
        return os.path.join(output_dir, f"{base_name}_face_{face_index + 1}.jpg")
