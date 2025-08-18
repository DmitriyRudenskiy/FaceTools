from src.core.interfaces.face_comparator import FaceComparator
import face_recognition
import os
from typing import List, Tuple


class FaceRecognitionFaceComparator(FaceComparator):
    """Сравнение лиц с использованием face_recognition"""

    def __init__(self):
        self.image_encodings = {}

    def load(self, directory_path):
        """Загружает изображения из директории"""
        print(f"Загружаю изображения из директории: {directory_path}")
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = [
            os.path.join(directory_path, filename)
            for filename in os.listdir(directory_path)
            if filename.lower().endswith(valid_extensions)
        ]
        self.image_encodings = {}
        sequential_index = 0
        for image_path in image_paths:
            try:
                # Сохраняем полный путь к изображению
                self.image_encodings[sequential_index] = {
                    'path': image_path,
                    'encoding': None
                }
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                if len(face_encodings) > 0 and face_encodings[0] is not None:
                    self.image_encodings[sequential_index]['encoding'] = face_encodings[0]
                    sequential_index += 1
                else:
                    print(f"Предупреждение: Не найдено лиц на изображении {os.path.basename(image_path)}")
                    # Удаляем запись, если лицо не найдено
                    del self.image_encodings[sequential_index]
            except Exception as e:
                print(f"Ошибка при обработке {os.path.basename(image_path)}: {e}")
                # Удаляем запись в случае ошибки
                if sequential_index in self.image_encodings:
                    del self.image_encodings[sequential_index]

    def compare(self, index1, index2):
        """Сравнивает два лица по индексам"""
        data1 = self.image_encodings.get(index1)
        data2 = self.image_encodings.get(index2)
        if data1 is None or data2 is None or data1['encoding'] is None or data2['encoding'] is None:
            # Если одно из изображений не имеет кодировки, считаем их несовпадающими
            return [False, 1.0]  # Максимальное расстояние
        encoding1 = data1['encoding']
        encoding2 = data2['encoding']
        results = face_recognition.compare_faces([encoding1], encoding2)
        face_distance = face_recognition.face_distance([encoding1], encoding2)
        return [results[0], face_distance[0]]

    def batch_compare(self, image_paths: list) -> list:
        """Сравнивает все изображения между собой"""
        n = len(image_paths)
        similarity_matrix = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = [True, 0.0]
                else:
                    similarity_matrix[i][j] = self.compare(i, j)
                    similarity_matrix[j][i] = similarity_matrix[i][j]  # Симметричная матрица
        return similarity_matrix