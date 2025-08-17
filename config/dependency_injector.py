from typing import Optional
from application.services import FaceProcessingService, FaceClusteringService
from infrastructure.image import OSImageLoader, ImagePreprocessor
from infrastructure.detection import InsightFaceDetector
from infrastructure.extraction import ArcFaceExtractor
from infrastructure.clustering import SilhouetteClusterAnalyzer, KMeansClusterer
from infrastructure.persistence import JSONResultSaver, FileSystemOrganizer
from core.interfaces import (
    ImageLoader, FaceDetector, FeatureExtractor,
    ClusterAnalyzer, Clusterer, ResultSaver, FileOrganizer
)


class DependencyInjector:
    """Фабрика для создания и инъекции зависимостей."""

    @staticmethod
    def create_face_processing_service(
            ctx_id: int = 0,
            det_size: Tuple[int, int] = (640, 640),
            det_thresh: float = 0.5,
            preprocessor=None
    ) -> FaceProcessingService:
        """Создает сервис обработки лиц с зависимостями."""
        image_loader = OSImageLoader(preprocessor=preprocessor)
        face_detector = InsightFaceDetector(ctx_id, det_size, det_thresh)
        feature_extractor = ArcFaceExtractor(ctx_id=ctx_id)

        return FaceProcessingService(image_loader, face_detector, feature_extractor)

    @staticmethod
    def create_face_clustering_service() -> FaceClusteringService:
        """Создает сервис кластеризации лиц с зависимостями."""
        cluster_analyzer = SilhouetteClusterAnalyzer()
        clusterer = KMeansClusterer()

        return FaceClusteringService(cluster_analyzer, clusterer)

    @staticmethod
    def create_result_saver() -> ResultSaver:
        """Создает сервис сохранения результатов."""
        return JSONResultSaver()

    @staticmethod
    def create_file_organizer() -> FileOrganizer:
        """Создает сервис организации файлов."""
        return FileSystemOrganizer()

    @staticmethod
    def create_default_pipeline(
            ctx_id: int = 0,
            det_size: Tuple[int, int] = (640, 640),
            det_thresh: float = 0.5
    ):
        """Создает полный пайплайн обработки с дефолтными зависимостями."""
        face_processing_service = DependencyInjector.create_face_processing_service(
            ctx_id, det_size, det_thresh
        )
        face_clustering_service = DependencyInjector.create_face_clustering_service()
        result_saver = DependencyInjector.create_result_saver()
        file_organizer = DependencyInjector.create_file_organizer()

        return FaceClusteringPipeline(
            face_processing_service,
            face_clustering_service,
            result_saver,
            file_organizer
        )