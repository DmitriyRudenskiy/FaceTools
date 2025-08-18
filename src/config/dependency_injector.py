
from src.application.services.face_extraction_service import FaceExtractionService # <<<--- НОВОЕ ИМЯ ФАЙЛА
from src.infrastructure.detection.yolo_detector import YOLOFaceDetector, DefaultBoundingBoxProcessor
from src.infrastructure.image.os_image_loader import OSImageLoader
from src.infrastructure.persistence.file_system_organizer import FileSystemOrganizer

class DependencyInjector:
    """Контейнер зависимостей для всего приложения"""
    def get_face_clustering_service(self) -> FaceExtractionService: # <<<--- Тип возвращаемого значения остался прежним, так как класс переименован
        """Создает и возвращает сервис извлечения лиц"""
        file_organizer = FileSystemOrganizer()
        return FaceExtractionService( # <<<--- Имя класса
            file_organizer=file_organizer,
            face_detector=YOLOFaceDetector(),
            bbox_processor=DefaultBoundingBoxProcessor(),
            image_loader=OSImageLoader(),
            result_saver=file_organizer
        )
