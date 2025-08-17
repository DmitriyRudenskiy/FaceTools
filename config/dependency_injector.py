from infrastructure.detection.yolo_detector import YOLOFaceDetector, DefaultBoundingBoxProcessor
from infrastructure.image.os_image_loader import OSImageLoader
from infrastructure.persistence.file_system_organizer import FileSystemOrganizer
from application.services.face_clustering_service import FaceClusteringService


class DependencyInjector:
    """Контейнер зависимостей для всего приложения"""

    def get_face_clustering_service(self) -> FaceClusteringService:
        """Создает и возвращает сервис кластеризации лиц"""
        file_organizer = FileSystemOrganizer()
        return FaceClusteringService(
            file_organizer=file_organizer,
            face_detector=YOLOFaceDetector(),  # Убедитесь, что имя класса верное
            bbox_processor=DefaultBoundingBoxProcessor(),
            image_loader=OSImageLoader(),
            result_saver=file_organizer
        )