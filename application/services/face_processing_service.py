from typing import List, Tuple, Dict
import numpy as np
from core.interfaces import ImageLoader, FaceDetector, FeatureExtractor
from domain.face import Face


class FaceProcessingService:
    """Сервис для обработки изображений с лицами."""

    def __init__(self,
                 image_loader: ImageLoader,
                 face_detector: FaceDetector,
                 feature_extractor: FeatureExtractor):
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