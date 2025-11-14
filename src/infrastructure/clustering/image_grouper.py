import time
from typing import List, Dict, Any, Optional
import numpy as np
import os

# Исправленный импорт класса Cluster
from src.domain.cluster import Cluster, ClusteringResult
from src.core.interfaces.clusterer import Clusterer
from sklearn.metrics import silhouette_score


class ImageGrouper:
    """Группировка изображений по схожести лиц с поддержкой различных алгоритмов кластеризации через внедрение зависимостей"""

    def __init__(self, similarity_matrix: List[List[float]], image_paths: List[str], clusterer: Clusterer):
        """Args:
        similarity_matrix: матрица схожести N x N
        image_paths: пути к изображениям
        clusterer: экземпляр Clusterer, реализующий алгоритм кластеризации
        """
        # Создаем копию матрицы, чтобы не изменять оригинальную
        self.similarity_matrix = [row[:] for row in similarity_matrix]
        self.image_paths = image_paths
        self.num_images = len(image_paths)
        self.clusterer = clusterer
        self.groups = []
        self.evaluation_metrics = {}
        # Валидация и коррекция матрицы
        self._validate_and_correct_matrix()

    def _validate_and_correct_matrix(self):
        """Проверяет и корректирует матрицу схожести"""
        # Проверка размерности
        if len(self.similarity_matrix) != self.num_images:
            raise ValueError("Размер матрицы схожести не совпадает с количеством изображений")

        # Проверка и коррекция
        for i in range(self.num_images):
            # Корректируем диагональные элементы
            if abs(self.similarity_matrix[i][i] - 1.0) > 1e-5:
                print(
                    f"Предупреждение: диагональный элемент [{i},{i}] должен быть равен 1.0, найдено {self.similarity_matrix[i][i]}. Исправляем на 1.0")
                self.similarity_matrix[i][i] = 1.0

            for j in range(i + 1, self.num_images):
                # Корректируем симметричность
                if abs(self.similarity_matrix[i][j] - self.similarity_matrix[j][i]) > 1e-5:
                    avg = (self.similarity_matrix[i][j] + self.similarity_matrix[j][i]) / 2
                    self.similarity_matrix[i][j] = avg
                    self.similarity_matrix[j][i] = avg
                    print(
                        f"Предупреждение: матрица не симметрична: элементы [{i},{j}] и [{j},{i}] отличаются. Используем среднее значение: {avg}")

                # Корректируем диапазон
                if self.similarity_matrix[i][j] < 0.0:
                    print(
                        f"Предупреждение: элемент [{i},{j}] меньше 0.0, найдено {self.similarity_matrix[i][j]}. Исправляем на 0.0")
                    self.similarity_matrix[i][j] = 0.0
                    self.similarity_matrix[j][i] = 0.0
                elif self.similarity_matrix[i][j] > 1.0:
                    print(
                        f"Предупреждение: элемент [{i},{j}] больше 1.0, найдено {self.similarity_matrix[i][j]}. Исправляем на 1.0")
                    self.similarity_matrix[i][j] = 1.0
                    self.similarity_matrix[j][i] = 1.0

    def _calculate_average_similarity(self, group_indices: List[int]) -> float:
        """Рассчитывает среднюю схожесть внутри группы"""
        total_similarity = 0.0
        count = 0
        for i_idx, i in enumerate(group_indices):
            for j in group_indices[i_idx + 1:]:
                similarity = self.similarity_matrix[i][j]
                total_similarity += similarity
                count += 1
        return total_similarity / count if count > 0 else 0.0

    def _evaluate_clustering(self, groups: List[List[int]]) -> Dict[str, Any]:
        """Оценивает качество кластеризации"""
        if not groups or len(groups) < 2:
            return {
                'silhouette_score': -1,
                'avg_cluster_similarity': 0,
                'cluster_sizes': [len(g) for g in groups]
            }

        # Подготовка меток для silhouette_score
        labels = np.full(self.num_images, -1)
        for cluster_id, group_indices in enumerate(groups):
            for idx in group_indices:
                labels[idx] = cluster_id

        # Вычисляем silhouette_score
        # Для этого нужно преобразовать матрицу схожести в матрицу расстояний
        distance_matrix = 1.0 - np.array(self.similarity_matrix)

        # silhouette_score требует минимум 2 кластера
        if len(set(labels)) >= 2:
            silhouette = silhouette_score(distance_matrix, labels, metric='precomputed')
        else:
            silhouette = -1

        # Средняя схожесть внутри кластеров
        avg_similarity = 0.0
        for group_indices in groups:
            avg_similarity += self._calculate_average_similarity(group_indices)
        avg_similarity /= len(groups)

        return {
            'silhouette_score': silhouette,
            'avg_cluster_similarity': avg_similarity,
            'cluster_sizes': [len(g) for g in groups]
        }

    def cluster(self) -> ClusteringResult:
        """Основной метод для выполнения кластеризации"""
        print(f"Начинаю группировку изображений методом '{self.clusterer.get_params()['method']}'...")

        # Выполняем кластеризацию
        start_time = time.time()
        self.groups = self.clusterer.cluster(self.similarity_matrix)
        clustering_time = time.time() - start_time
        print(f"Кластеризация завершена за {clustering_time:.2f} секунд")

        # Оценка качества кластеризации
        self.evaluation_metrics = self._evaluate_clustering(self.groups)
        print(f"Качество кластеризации: силуэтный коэффициент = {self.evaluation_metrics['silhouette_score']:.4f}, "
              f"средняя схожесть = {self.evaluation_metrics['avg_cluster_similarity']:.4f}")
        print(f"Размеры кластеров: {self.evaluation_metrics['cluster_sizes']}")

        # Сортируем группы по размеру (от большей к меньшей)
        self.groups.sort(key=len, reverse=True)

        # Подготовка данных для возврата
        groups_data = []
        for i, group_indices in enumerate(self.groups):
            # Рассчитываем средние расстояния внутри группы
            avg_similarity = self._calculate_average_similarity(group_indices)

            # Создаем объект Cluster (исправлено: добавлен импорт)
            group_data = Cluster(
                id=i,
                size=len(group_indices),
                representative=self.image_paths[group_indices[0]],
                representative_path=self.image_paths[group_indices[0]],
                members=[self.image_paths[idx] for idx in group_indices],
                members_paths=[self.image_paths[idx] for idx in group_indices],
                average_similarity=avg_similarity
            )
            groups_data.append(group_data)

        # Подготавливаем нераспознанные изображения
        all_indices = set(range(self.num_images))
        used_indices_in_groups = set()
        for group in groups_data:
            for path in group.members_paths:
                try:
                    idx = self.image_paths.index(path)
                    used_indices_in_groups.add(idx)
                except ValueError:
                    continue

        unrecognized_indices = all_indices - used_indices_in_groups
        unrecognized_images = [
            {"filename": os.path.basename(self.image_paths[idx]),
             "full_path": self.image_paths[idx]}
            for idx in unrecognized_indices
        ]

        # Возвращаем результат кластеризации
        return ClusteringResult(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            total_clusters=len(groups_data),
            unrecognized_count=len(unrecognized_images),
            clusters=groups_data,
            unrecognized_images=unrecognized_images,
            clustering_params=self.clusterer.get_params(),
            evaluation_metrics=self.evaluation_metrics
        )