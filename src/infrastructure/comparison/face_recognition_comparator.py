from src.core.interfaces.face_comparator import FaceComparator
import face_recognition
import os
from typing import List, Tuple


class FaceRecognitionFaceComparator(FaceComparator):
    """Сравнение лиц с использованием face_recognition"""

    def __init__(self, tolerance: float = 0.6):
        self.tolerance = tolerance
        self.face_encodings = {}
        self.image_paths = []

    def load_images(self, directory_path: str) -> None:
        """Загружает изображения из директории"""
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.image_paths = [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.lower().endswith(valid_extensions)
        ]

        self.face_encodings = {}
        for i, image_path in enumerate(self.image_paths):
            try:
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    self.face_encodings[i] = face_encodings[0]
            except Exception as e:
                print(f"Ошибка при обработке {os.path.basename(image_path)}: {e}")

    def compare(self, idx1: int, idx2: int) -> Tuple[bool, float]:
        """Сравнивает два лица по индексам"""
        if idx1 not in self.face_encodings or idx2 not in self.face_encodings:
            return False, 1.0

        encoding1 = self.face_encodings[idx1]
        encoding2 = self.face_encodings[idx2]
        results = face_recognition.compare_faces([encoding1], encoding2, tolerance=self.tolerance)
        distance = face_recognition.face_distance([encoding1], encoding2)[0]
        return results[0], distance

    def batch_compare(self) -> List[List[Tuple[bool, float]]]:
        """Сравнивает все изображения между собой"""
        n = len(self.image_paths)
        similarity_matrix = [[None] * n for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = (True, 0.0)
                else:
                    similarity_matrix[i][j] = self.compare(i, j)
                    similarity_matrix[j][i] = similarity_matrix[i][j]  # Симметричная матрица

        return similarity_matrix