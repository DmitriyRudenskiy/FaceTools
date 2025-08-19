# src/application/use_cases/cluster_faces.py
import time
from typing import Any, Dict, List, Optional

from src.application.services import FaceClusteringService, FaceProcessingService
from src.core.interfaces import FileOrganizer, ResultSaver

# Используем абсолютные импорты, как в остальной части проекта
from src.domain.cluster import ClusteringResult


class ClusterFacesUseCase:
    """Use case для кластеризации лиц на изображениях."""

    def __init__(
        self,
        face_processing_service: FaceProcessingService,
        face_clustering_service: FaceClusteringService,
        result_saver: ResultSaver,
        file_organizer: FileOrganizer,
    ):
        self.face_processing_service = face_processing_service
        self.face_clustering_service = face_clustering_service
        self.result_saver = result_saver
        self.file_organizer = file_organizer
        self.similarity_matrix = None
        self.faces: List = []  # Явно указываем тип
        self.labels: List = []  # Явно указываем тип

    def execute(
        self,
        input_dir: str,
        output_json: bool = True,
        output_json_path: str = "groups.json",
        organize_files: bool = False,
        dest_dir: Optional[str] = None,
        max_clusters: int = 20,
        method: str = "silhouette",
    ) -> Dict[str, Any]:
        """Выполняет полный процесс кластеризации."""
        start_time = time.time()

        # 1. Обработка изображений и извлечение лиц
        self.faces, _ = self.face_processing_service.process_directory(input_dir)

        if len(self.faces) < 2:
            return {
                "total_faces": len(self.faces),
                "total_groups": 0,
                "elapsed_time": time.time() - start_time,
            }

        # 2. Кластеризация лиц
        optimal_k, self.labels = self.face_clustering_service.cluster_faces(
            self.faces, max_clusters, method
        )

        # 3. Подготовка и сохранение результатов
        # Передаем необходимые данные в ClusteringResult
        result = ClusteringResult(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            total_clusters=optimal_k if optimal_k > 0 else 0,
            unrecognized_count=0,  # Этот параметр требует доработки логики
            clusters=[],  # Этот параметр требует доработки логики для заполнения
            unrecognized_images=[],  # Этот параметр требует доработки логики
        )

        if output_json:
            self.result_saver.save(result, output_json_path)

        if organize_files and dest_dir:
            self.file_organizer.organize_by_clusters(result.clusters, dest_dir)

        return {
            "total_faces": len(self.faces),
            "total_groups": optimal_k,
            "elapsed_time": time.time() - start_time,
            # "clustering_result": result # Можно добавить для отладки
        }

    @classmethod
    def with_default_dependencies(
        cls, ctx_id: int = 0, det_size: tuple = (640, 640), det_thresh: float = 0.5
    ):
        """Создает экземпляр use case с дефолтными зависимостями."""
        # Используем абсолютный импорт для DependencyInjector
        from src.config.dependency_injector import DependencyInjector

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
            file_organizer,
        )
