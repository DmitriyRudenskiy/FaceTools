import numpy as np
from deepface import DeepFace

class ExpertDeepFaceAndArcFace:
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
                    enforce_detection=True
                )

                # Берем первый найденный эмбеддинг (первое лицо на изображении)
                embedding = embedding_objs[0]["embedding"]
                embedding = np.array(embedding)

                self.storage.append((embedding, image_path))
            except Exception as e:
                print(f"Ошибка при обработке {image_path}: {e}")

    def compare(self, index1, index2):
        """Сравнение двух изображений по индексам в хранилище"""
        emb1 = self.storage[index1][0]
        emb2 = self.storage[index2][0]

        # Проверка на наличие эмбеддингов
        if emb1 is None or emb2 is None:
            return float('inf')  # Возвращаем максимальное расстояние если одно из лиц не найдено

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
            "similarity": similarity
        }

    def get_image_path(self, index):
        """Получение пути к изображению по индексу"""
        return self.storage[index][1]