import sys
import os

# Добавляем путь к src директории
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.expert_deepface import ExpertDeepFaceAndArcFace
from utils.compare_matrix import CompareMatrix

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