from typing import Dict, Any, Optional
import time
from domain.result import ClusteringResult
from application.services import FaceProcessingService, FaceClusteringService
from core.interfaces import ResultSaver, FileOrganizer


class ClusterFacesUseCase:
    """Use case для кластеризации лиц на изображениях."""

    def __init__(self,
                 face_processing_service: FaceProcessingService,
                 face_clustering_service: FaceClusteringService,
                 result_saver: ResultSaver,
                 file_organizer: FileOrganizer):
        self.face_processing_service = face_processing_service
        self.face_clustering_service = face_clustering_service
        self.result_saver = result_saver
        self.file_organizer = file_organizer
        self.similarity_matrix = None
        self.faces = []
        self.labels = []

    def execute(self,
                input_dir: str,
                output_json: bool = True,
                output_json_path: str = "groups.json",
                organize_files: bool = False,
                dest_dir: Optional[str] = None,
                max_clusters: int = 20,
                method: str = 'silhouette') -> Dict[str, Any]:
        """Выполняет полный процесс кластеризации."""
        start_time = time.time()

        # 1. Обработка изображений и извлечение лиц
        self.faces, _ = self.face_processing_service.process_directory(input_dir)

        if len(self.faces) < 2:
            return {
                "total_faces": len(self.faces),
                "total_groups": 0,
                "elapsed_time": time.time() - start_time
            }

        # 2. Кластеризация лиц
        optimal_k, self.labels = self.face_clustering_service.cluster_faces(
            self.faces, max_clusters, method
        )

        # 3. Подготовка и сохранение результатов
        result = ClusteringResult(
            faces=self.faces,
            labels=self.labels,
            optimal_k=optimal_k,
            input_dir=input_dir
        )

        if output_json:
            self.result_saver.save(result, output_json_path)

        # 4. Организация файлов по группам (если требуется)
        if organize_files and dest_dir:
            self.file_organizer.organize(result, dest_dir)

        elapsed_time = time.time() - start_time
        return {
            "total_faces": len(self.faces),
            "total_groups": optimal_k,
            "elapsed_time": elapsed_time
        }

    def display_similarity_matrix(self):
        """Отображает матрицу схожести."""
        # Логика отображения матрицы
        pass

    def display_reference_table(self):
        """Отображает таблицу сопоставления с эталонами."""
        # Логика отображения таблицы
        pass

    @classmethod
    def create_default(cls, ctx_id: int = 0, det_size: tuple = (640, 640), det_thresh: float = 0.5):
        """Создает экземпляр use case с дефолтными зависимостями."""
        face_processing_service = DependencyInjector.create_face_processing_service(
            ctx_id, det_size, det_thresh
        )
        face_clustering_service = DependencyInjector.create_face_clustering_service()
        result_saver = DependencyInjector.create_result_saver()
        file_organizer = DependencyInjector.create_file_organizer()

        return cls(
            face_processing_service,
            face_clustering_service,
            result_saver,
            file_organizer
        )