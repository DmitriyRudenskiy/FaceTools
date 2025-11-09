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


class HierarchicalClusterer:
    """Иерархическая кластеризация с динамической остановкой"""

    def __init__(self, min_cluster_size: int = 2, linkage_method: str = 'average',
                 inconsistency_threshold: float = 1.5):
        """
        Args:
            min_cluster_size: минимальный размер кластера
            linkage_method: метод связности ('single', 'complete', 'average', 'ward')
            inconsistency_threshold: порог коэффициента несогласованности
        """
        self.min_cluster_size = min_cluster_size
        self.linkage_method = linkage_method
        self.inconsistency_threshold = inconsistency_threshold
        self.cut_height = None

    def cluster(self, similarity_matrix: List[List[float]]) -> List[List[int]]:
        n = len(similarity_matrix)

        # Преобразуем схожесть в расстояние: D = 1 - S
        distance_matrix = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    distance_matrix[i][j] = 0.0
                else:
                    distance_matrix[i][j] = 1.0 - similarity_matrix[i][j]

        # Преобразуем в condensed form для иерархической кластеризации
        condensed_dist = squareform(distance_matrix, checks=False)

        # Выполняем агломеративную кластеризацию
        Z = linkage(condensed_dist, method=self.linkage_method)

        # Вычисляем коэффициент несогласованности
        incons = inconsistent(Z)
        # Используем последний столбец (коэффициент несогласованности)
        incons_vals = incons[:, -1]

        # Определяем порог остановки
        cut_height = None
        for i in range(len(Z) - 1, -1, -1):
            if incons_vals[i] < self.inconsistency_threshold:
                cut_height = Z[i, 2]  # высота на этом уровне
                break

        # Если не найдено подходящего уровня, используем медиану высот
        if cut_height is None:
            cut_height = np.median(Z[:, 2])

        self.cut_height = cut_height

        # Получаем метки кластеров
        labels = fcluster(Z, cut_height, criterion='distance')

        # Группируем индексы по меткам
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        # Фильтрация по минимальному размеру
        filtered_clusters = [comp for comp in clusters.values() if len(comp) >= self.min_cluster_size]
        return filtered_clusters

    def get_params(self) -> Dict:
        return {
            'method': 'hierarchical',
            'linkage_method': self.linkage_method,
            'inconsistency_threshold': self.inconsistency_threshold,
            'cut_height': self.cut_height,
            'min_cluster_size': self.min_cluster_size
        }