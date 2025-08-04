import numpy as np
import torch
from deepface import DeepFace
from PIL import Image
import cv2


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


# Класс для работы с матрицей сравнений (остается без изменений)
class CompareMatrix:
    def __init__(self, size):
        self.matrix = np.zeros((size, size))

    def fill(self, comparator):
        """Заполнение матрицы: диагональ - NULL, заполняются только элементы i > j"""
        size = self.matrix.shape[0]
        for i in range(size):
            for j in range(size):
                if i == j:
                    self.matrix[i, j] = np.nan
                elif i > j:
                    self.matrix[i, j] = comparator.compare(i, j)
                    self.matrix[j, i] = self.matrix[i, j]

    def display(self):
        """Вывод матрицы на экран"""
        print("Текущая матрица:")
        print(self.matrix)


# Пример использования:
if __name__ == "__main__":
    # Создаем экземпляр компаратора
    comparator = ExpertDeepFaceAndArcFace(
        detector_backend="retinaface",
        distance_metric="cosine"
    )

    # Инициализируем с изображениями
    image_paths = [
        "/Users/user/__!make_face/refer_bibi_1.png",
        "/Users/user/__!make_face/refer_bibi_2.png",
        "/Users/user/__!make_face/frame_Bibi/frame_05719_face_1.jpg",
        "/Users/user/Downloads/ideogram_download_2025-08-03/0001_1_a-striking-artistic-portrait-of-a-woman-_55Pb9rG2QAePxTT6mPxVVA_niCnd0CrQhWLT9HfAwcBwA.jpeg",
        "/Users/user/Downloads/ideogram_download_2025-08-03/0003_4_a-captivating-beauty-portrait-of-a-strik_zy28OHyKTIWGe3DduVimeA_yVxFlr85TM-wGt_i9yt04w.jpeg"
    ]

    comparator.init(image_paths)
    print(f"Количество обработанных изображений: {len(comparator.storage)}")

    # Сравниваем первые два изображения
    if len(comparator.storage) >= 2:
        distance = comparator.compare(0, 1)
        print(f"Расстояние между первыми двумя изображениями: {distance}")

        # Верификация
        result = comparator.verify(distance)
        print(f"Верификация: {result}")

    # Создаем и заполняем матрицу сравнений
    if len(comparator.storage) > 0:
        matrix = CompareMatrix(len(comparator.storage))
        matrix.fill(comparator)
        matrix.display()