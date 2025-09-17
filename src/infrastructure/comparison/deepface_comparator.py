from typing import List, Tuple, Dict, Any
import numpy as np
from deepface import DeepFace
import cv2


class FaceEmbeddingStorage:
    """Класс для хранения и управления эмбеддингами лиц"""

    def __init__(self):
        self.storage = []  # type: List[Tuple[np.ndarray, str]]

    def _find_index_by_path(self, path: str) -> int:
        """Находит индекс изображения в хранилище по пути"""
        for i, (_, stored_path) in enumerate(self.storage):
            if stored_path == path:
                return i
        return -1

    def compare_by_index(self, index1: int, index2: int) -> float:
        """Сравнение двух изображений по индексам в хранилище"""
        emb1 = self.storage[index1][0]
        emb2 = self.storage[index2][0]

        if emb1 is None or emb2 is None:
            return float('inf')

        dot_product = np.dot(emb1, emb2)
        norm_emb1 = np.linalg.norm(emb1)
        norm_emb2 = np.linalg.norm(emb2)
        similarity = dot_product / (norm_emb1 * norm_emb2)
        distance = 1 - similarity

        return distance

    def get_face_count(self) -> int:
        """Возвращает количество лиц в хранилище"""
        return len(self.storage)

    def add_to_storage(self, embedding: np.ndarray, image_path: str) -> None:
        """Добавляет эмбеддинг лица в хранилище"""
        self.storage.append((embedding, image_path))


class DeepFaceComparator:
    """Система распознавания лиц с использованием DeepFace"""

    def __init__(self):
        self.storage = FaceEmbeddingStorage()
        self.detector_backend = "retinaface"
        self.distance_metric = "cosine"
        self.model_name = "ArcFace"

    def init(self, image_paths: List[str]) -> List[str]:
        """Инициализация хранилища эмбеддингов для списка изображений"""
        successful_paths = []  # Список успешно обработанных путей

        for image_path in image_paths:
            try:
                embedding_objs = DeepFace.represent(
                    img_path=image_path,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=True,
                )

                embedding = embedding_objs[0]["embedding"]
                embedding = np.array(embedding)

                self.storage.add_to_storage(embedding, image_path)
                successful_paths.append(image_path)  # Добавляем в список успешных
            except Exception as e:
                print(f"Ошибка при обработке {image_path}")

        return successful_paths  # Возвращаем только успешно обработанные файлы

    def compare_faces(self, face1_path: str, face2_path: str) -> Tuple[bool, float]:
        """Сравнивает два лица и возвращает (совпадение, расстояние)"""
        idx1 = self.storage._find_index_by_path(face1_path)
        idx2 = self.storage._find_index_by_path(face2_path)

        if idx1 == -1 or idx2 == -1:
            return (False, 1.0)

        distance = self.storage.compare_by_index(idx1, idx2)
        result = self.verify_similarity(distance)
        return (result["verified"], distance)

    def verify_similarity(self, distance: float) -> dict:
        """Проверяет, являются ли лица одним человеком на основе расстояния"""
        threshold = 0.2557
        is_verified = distance < threshold
        similarity = 1 - distance

        return {
            "verified": is_verified,
            "distance": distance,
            "threshold": threshold,
            "similarity": similarity,
        }

    def detect_faces(self, image_path: str, extract_embeddings: bool = False) -> List[Dict[str, Any]]:
        """
        Обнаруживает лица на изображении и возвращает информацию о них

        Args:
            image_path: путь к изображению
            extract_embeddings: извлекать ли эмбеддинги лиц

        Returns:
            Список словарей с информацией о найденных лицах
        """
        try:
            # Загружаем изображение
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")

            # Используем DeepFace для обнаружения лиц
            face_objs = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,  # Не вызывать исключение, если лица не найдены
            )

            result = []
            for i, face_obj in enumerate(face_objs):
                face_info = {
                    "face_index": i,
                    "facial_area": face_obj.get("facial_area", {}),
                    "confidence": face_obj.get("face_confidence", 1.0),
                }

                # Если запрошено извлечение эмбеддингов
                if extract_embeddings:
                    face_info["embedding"] = np.array(face_obj["embedding"])

                result.append(face_info)

            return result

        except Exception as e:
            print(f"Ошибка при обнаружении лиц на изображении {image_path}: {e}")
            return []
