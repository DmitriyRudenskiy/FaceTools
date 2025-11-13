from typing import List, Tuple, Dict, Any
import numpy as np
from deepface import DeepFace
import cv2
import os
import re
import tempfile
import shutil
from pathlib import Path

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
        """Сравнение двух изображений по индексам в хранилище

        Возвращает меру схожести в диапазоне [0, 1], где:
        - 1.0 = полное совпадение (евклидово расстояние = 0)
        - 0.0 = максимальное несовпадение (евклидово расстояние = 2)

        Используется преобразование: (cosθ + 1.0) / 2.0
        """
        emb1 = self.storage[index1][0]
        emb2 = self.storage[index2][0]

        if emb1 is None or emb2 is None:
            return 0.0  # Теперь возвращаем минимальную схожесть вместо inf

        # Вычисляем косинусное сходство (cosθ)
        dot_product = np.dot(emb1, emb2)
        norm_emb1 = np.linalg.norm(emb1)
        norm_emb2 = np.linalg.norm(emb2)

        # Защита от деления на ноль
        if norm_emb1 == 0 or norm_emb2 == 0:
            return 0.0

        cos_theta = dot_product / (norm_emb1 * norm_emb2)

        # Преобразуем в меру схожести [0, 1]
        similarity = (cos_theta + 1.0) / 2.0

        # Гарантируем корректный диапазон значений
        return max(0.0, min(1.0, similarity))

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

    def sanitize_path(self, path): 
        """Проверяет путь на наличие неанглийских символов и копирует файл во временную папку при необходимости."""
        if re.search(r'[^\x00-\x7F]', str(path)):
            # Создаем временный файл с английским именем
            temp_dir = tempfile.mkdtemp()
            filename = Path(path).name
            temp_path = os.path.join(temp_dir, filename)

            # Копируем файл во временную папку
            shutil.copy2(path, temp_path)
            return temp_path, temp_dir
        return path, None

    def init(self, image_paths: List[str]) -> List[str]:
        """Инициализация хранилища эмбеддингов для списка изображений"""
        successful_paths = []  # Список успешно обработанных путей

        for image_path in image_paths:
            sanitized_path, temp_dir = self.sanitize_path(image_path)
            if temp_dir:
                image_path = sanitized_path

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
                print(f"\n\nОшибка при обработке:\n\t{os.path.basename(image_path)}:\n\t{e}")

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
