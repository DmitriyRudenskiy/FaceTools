from src.core.interfaces.clusterer import Clusterer
from src.domain.cluster import Cluster, ClusteringResult
from src.infrastructure.comparison.face_recognition_comparator import FaceRecognitionFaceComparator
import os
import time
from typing import List


class ImageGrouper(Clusterer):
    """Группировка изображений по схожести лиц"""

    def __init__(self, comparator: FaceRecognitionFaceComparator):
        self.comparator = comparator

    def cluster(self, image_paths: List[str]) -> ClusteringResult:
        """Группирует изображения по схожести лиц"""
        start_time = time.time()
        print("Начинаю группировку изображений...")

        # Создаем матрицу схожести
        self.comparator.load_images(os.path.dirname(image_paths[0]) if image_paths else "")
        similarity_matrix = self.comparator.batch_compare()

        # Группируем изображения
        n = len(image_paths)
        groups = []
        used_indices = set()

        for i in range(n):
            if i in used_indices or i not in self.comparator.face_encodings:
                continue

            current_group = [i]
            used_indices.add(i)

            for j in range(i + 1, n):
                if j in used_indices or j not in self.comparator.face_encodings:
                    continue

                if similarity_matrix[i][j][0]:  # Если лица совпадают
                    current_group.append(j)
                    used_indices.add(j)

            if len(current_group) > 1:
                groups.append(current_group)

        # Подготовка результата
        clusters = []
        for i, group_indices in enumerate(groups):
            # Вычисляем среднее расстояние для каждого изображения в группе
            distances = []
            for idx in group_indices:
                total_distance = 0.0
                count = 0
                for other_idx in group_indices:
                    if idx != other_idx:
                        total_distance += similarity_matrix[idx][other_idx][1]
                        count += 1
                avg_distance = total_distance / count if count > 0 else 0.0
                distances.append((avg_distance, idx))

            # Находим представителя группы (с минимальным средним расстоянием)
            min_distance, rep_idx = min(distances, key=lambda x: x[0])
            representative = os.path.basename(self.comparator.image_paths[rep_idx])

            # Создаем объект кластера
            clusters.append(Cluster(
                id=i + 1,
                size=len(group_indices),
                representative=representative,
                representative_path=self.comparator.image_paths[rep_idx],
                members=[os.path.basename(self.comparator.image_paths[idx]) for idx in group_indices],
                members_paths=[self.comparator.image_paths[idx] for idx in group_indices],
                average_similarity=1 - min_distance
            ))

        # Определяем нераспознанные изображения
        all_indices = set(range(n))
        used_indices = set(idx for group in groups for idx in group)
        unrecognized_indices = all_indices - used_indices
        unrecognized_images = [
            {"filename": os.path.basename(self.comparator.image_paths[idx]),
             "full_path": self.comparator.image_paths[idx]}
            for idx in unrecognized_indices
        ]

        end_time = time.time()
        print(f"Группировка завершена за {end_time - start_time:.2f} секунд")

        return ClusteringResult(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            total_clusters=len(clusters),
            unrecognized_count=len(unrecognized_images),
            clusters=clusters,
            unrecognized_images=unrecognized_images
        )