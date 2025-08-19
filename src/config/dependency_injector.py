from src.application.services.face_crop_service import FaceCropService
from src.application.services.face_detection_service import \
    FaceDetectionService
from src.infrastructure.clustering.legacy_image_grouper import ImageGrouper
from src.infrastructure.clustering.reference_table_printer import \
    ReferenceTablePrinter
from src.infrastructure.comparison.deepface_comparator import \
    DeepFaceFaceComparator
from src.infrastructure.detection.yolo_detector import (
    DefaultBoundingBoxProcessor, YOLOFaceDetector)
from src.infrastructure.image.os_image_loader import OSImageLoader
from src.infrastructure.persistence.file_system_organizer import \
    FileSystemOrganizer
from src.infrastructure.persistence.group_organizer import GroupOrganizer


class DependencyInjector:
    """Контейнер зависимостей для всего приложения"""

    def get_face_detection_service(
        self,
    ) -> FaceDetectionService:
        """Создает и возвращает сервис извлечения лиц"""
        file_organizer = FileSystemOrganizer()
        return FaceDetectionService(  # <<<--- Имя класса
            file_organizer=file_organizer,
            face_detector=YOLOFaceDetector(),
            bbox_processor=DefaultBoundingBoxProcessor(),
            image_loader=OSImageLoader(),
            result_saver=file_organizer,
        )

    def get_image_grouper(self, similarity_matrix, image_paths):
        """Создает и возвращает ImageGrouper"""
        return ImageGrouper(similarity_matrix, image_paths)

    def get_reference_table_printer(self, similarity_matrix, image_paths, num_images):
        """Создает и возвращает ReferenceTablePrinter"""
        return ReferenceTablePrinter(similarity_matrix, image_paths, num_images)

    def get_group_organizer(self, groups_data, destination_directory):
        """Создает и возвращает GroupOrganizer"""
        return GroupOrganizer(groups_data, destination_directory)

    # В класс DependencyInjector добавляем новый метод:
    def get_deepface_face_comparator(self) -> DeepFaceFaceComparator:
        """Создает и возвращает компаратор на основе DeepFace"""
        from src.infrastructure.comparison.deepface_comparator import \
            DeepFaceFaceComparator

        return DeepFaceFaceComparator()

    def get_face_crop_service(self):
        """Создает и возвращает сервис вырезания лиц"""
        # Создаем зависимости
        file_organizer = FileSystemOrganizer()
        face_detector = YOLOFaceDetector()
        bbox_processor = DefaultBoundingBoxProcessor()
        image_loader = OSImageLoader()

        # Возвращаем экземпляр сервиса
        return FaceCropService(
            file_organizer=file_organizer,
            face_detector=face_detector,
            bbox_processor=bbox_processor,
            image_loader=image_loader,
        )
