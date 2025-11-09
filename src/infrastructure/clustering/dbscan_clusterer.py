import os
import time
from typing import List, Dict, Tuple, Optional, Protocol
import numpy as np
from abc import ABC, abstractmethod
from scipy.cluster.hierarchy import linkage, fcluster, inconsistent
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score

from src.domain.cluster import Cluster, ClusteringResult

class DBSCANClusterer:
    """DBSCAN с адаптивным eps и обработкой шума"""

    def __init__(self, min_cluster_size: int = 2, eps: Optional[float] = None,
                 k_for_eps: Optional[int] = None):
        """
        Args:
            min_cluster_size: минимальный размер кластера (соответствует min_samples в DBSCAN)
            eps: параметр eps (если None, определяется автоматически)
            k_for_eps: количество соседей для определения eps (если None, используется min_cluster_size)
        """
        self.min_cluster_size = min_cluster_size
        self.eps = eps
        self.k_for_eps = k_for_eps
        self.k_distances = None

    def cluster(self, similarity_matrix: List[List[float]]) -> List[List[int]]:
        n = len(similarity_matrix)

        # Преобразуем схожесть в расстояние
        distance_matrix = 1 - np.array(similarity_matrix)
        np.fill_diagonal(distance_matrix, 0)

        # Определяем eps через k-расстояния
        if self.eps is None:
            # Используем k = min_cluster_size или k_for_eps
            k = self.k_for_eps if self.k_for_eps is not None else max(2, self.min_cluster_size)
            k = min(k, n - 1)  # не может быть больше количества точек - 1

            k_distances = []
            for i in range(n):
                distances = sorted(distance_matrix[i])
                k_distances.append(distances[k] if k < len(distances) else distances[-1])

            # Сохраняем для анализа
            self.k_distances = k_distances

            # Находим "колено" в графике k-расстояний
            k_distances_sorted = sorted(k_distances)
            if len(k_distances_sorted) > 1:
                gaps = [k_distances_sorted[i + 1] - k_distances_sorted[i] for i in range(len(k_distances_sorted) - 1)]
                max_gap_idx = np.argmax(gaps)
                self.eps = k_distances_sorted[max_gap_idx]
            else:
                self.eps = k_distances_sorted[0]

        # Запускаем DBSCAN
        db = DBSCAN(eps=self.eps, min_samples=self.min_cluster_size, metric='precomputed')
        labels = db.fit_predict(distance_matrix)

        # Обработка шумовых точек (помеченные как -1)
        # Назначаем шумовые точки к ближайшему кластеру
        noise_indices = np.where(labels == -1)[0]
        if len(noise_indices) > 0:
            for idx in noise_indices:
                # Находим ближайший кластер по средней схожести
                max_similarity = -1
                best_cluster = None

                for cluster_id in set(labels):
                    if cluster_id == -1:
                        continue

                    cluster_indices = np.where(labels == cluster_id)[0]
                    avg_similarity = np.mean([similarity_matrix[idx][j] for j in cluster_indices])

                    if avg_similarity > max_similarity:
                        max_similarity = avg_similarity
                        best_cluster = cluster_id

                # Назначаем шумовую точку в лучший кластер, если схожесть достаточно высока
                if best_cluster is not None and max_similarity > 0.5:
                    labels[idx] = best_cluster

        # Группируем индексы по меткам
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        # Удаляем шумовые кластеры (если они остались)
        if -1 in clusters:
            del clusters[-1]

        # Фильтрация по минимальному размеру (уже учтено в DBSCAN, но проверяем)
        filtered_clusters = [comp for comp in clusters.values() if len(comp) >= self.min_cluster_size]
        return filtered_clusters

    def get_params(self) -> Dict:
        return {
            'method': 'dbscan',
            'eps': self.eps,
            'min_cluster_size': self.min_cluster_size,
            'k_for_eps': self.k_for_eps
        }