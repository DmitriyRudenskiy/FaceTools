# src/infrastructure/comparison/deepface_comparator.py
from typing import List, Tuple

import numpy as np
from deepface import DeepFace


class DeepFaceFaceComparator:
    """Сравнение лиц с использованием DeepFace и ArcFace"""

    def __init__(self, detector_backend="retinaface", distance_metric="cosine"):
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        self.model_name = "ArcFace"
        self.storage = []

    def init(self, image_paths):
        """Инициализация хранилища эмбеддингов для списка изображений"""
        for image_path in image_paths:
            try:
                # Получаем эмбеддинг через DeepFace с моделью ArcFace
                embedding_objs = DeepFace.represent(
                    img_path=image_path,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=True,
                )

                # Берем первый найденный эмбеддинг (первое лицо на изображении)
                embedding = embedding_objs[0]["embedding"]
                embedding = np.array(embedding)

                self.storage.append((embedding, image_path))
            except Exception as e:
                print(f"Ошибка при обработке {image_path}: {e}")

    def compare(self, face1_path: str, face2_path: str) -> Tuple[bool, float]:
        """Сравнивает два лица и возвращает (совпадение, расстояние)"""
        # Нужно найти индексы в хранилище
        idx1 = self._find_index_by_path(face1_path)
        idx2 = self._find_index_by_path(face2_path)

        if idx1 == -1 or idx2 == -1:
            return (False, 1.0)  # Максимальное расстояние

        distance = self._compare_by_index(idx1, idx2)
        result = self.verify(distance)
        return (result["verified"], distance)

    def _find_index_by_path(self, path: str) -> int:
        """Находит индекс изображения в хранилище по пути"""
        for i, (_, stored_path) in enumerate(self.storage):
            if stored_path == path:
                return i
        return -1

    def _compare_by_index(self, index1, index2):
        """Сравнение двух изображений по индексам в хранилище"""
        emb1 = self.storage[index1][0]
        emb2 = self.storage[index2][0]

        # Проверка на наличие эмбеддингов
        if emb1 is None or emb2 is None:
            return float(
                "inf"
            )  # Возвращаем максимальное расстояние если одно из лиц не найдено

        # По умолчанию используем косинусное расстояние
        dot_product = np.dot(emb1, emb2)
        norm_emb1 = np.linalg.norm(emb1)
        norm_emb2 = np.linalg.norm(emb2)
        similarity = dot_product / (norm_emb1 * norm_emb2)
        distance = 1 - similarity

        return distance

    def verify(self, distance):
        threshold = 0.2557

        is_verified = distance < threshold
        similarity = 1 - distance

        return {
            "verified": is_verified,
            "distance": distance,
            "threshold": threshold,
            "similarity": similarity,
        }

    def batch_compare(self, image_paths: List[str]) -> list:
        """Сравнивает все изображения между собой"""
        n = len(image_paths)
        similarity_matrix = [[None] * n for _ in range(n)]

        # Инициализируем хранилище
        self.init(image_paths)

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = (True, 0.0)
                else:
                    distance = self._compare_by_index(i, j)
                    result = self.verify(distance)
                    similarity_matrix[i][j] = (result["verified"], distance)
                    similarity_matrix[j][i] = similarity_matrix[i][
                        j
                    ]  # Симметричная матрица
        return similarity_matrix
