from typing import Any, List, Tuple

import numpy as np

from src.core.interfaces import FaceDetector, FileOrganizer, ImageLoader
from src.domain.face import Face


class FaceProcessingService:
    def __init__(
        self,
        file_organizer: FileOrganizer,
        face_detector: FaceDetector,
        image_loader: ImageLoader,
        # Добавляем feature_extractor в конструктор
        feature_extractor: Any,  # Или FeatureExtractor, если он определен
    ):
        self.image_loader = image_loader
        self.face_detector = face_detector
        self.feature_extractor = feature_extractor

    def process_directory(self, directory: str) -> Tuple[List[Face], List[np.ndarray]]:
        """Обрабатывает все изображения в указанной директории."""
        # Загрузка изображений
        images = self.image_loader.load_images(directory)

        faces = []
        embeddings = []

        # Обработка каждого изображения
        for image_path, image in images:
            # Детекция лиц
            detected_faces = self.face_detector.detect_faces(image)

            # Извлечение признаков для каждого лица
            for face in detected_faces:
                embedding = self.feature_extractor.extract_features(image, face)
                face.embedding = embedding
                face.image_path = image_path

                faces.append(face)
                embeddings.append(embedding)

        return faces, embeddings
