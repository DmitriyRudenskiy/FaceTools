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

class SpectralClusterer:
    """Спектральная кластеризация с эвристикой eigengap"""

    def __init__(self, min_cluster_size: int = 2, max_clusters: Optional[int] = None):
        """
        Args:
            min_cluster_size: минимальный размер кластера
            max_clusters: максимальное количество кластеров (если None, определяется автоматически)
        """
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.optimal_k = None

    def cluster(self, similarity_matrix: List[List[float]]) -> List[List[int]]:
        n = len(similarity_matrix)

        # Построение матрицы смежности (используем схожесть напрямую)
        adjacency = np.array(similarity_matrix)
        np.fill_diagonal(adjacency, 0)  # убираем диагональ

        # Нормализованный графовый Лапласиан: L = I - D^(-1/2) A D^(-1/2)
        D = np.diag(np.sum(adjacency, axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        L = np.eye(n) - D_inv_sqrt @ adjacency @ D_inv_sqrt

        # Вычисляем собственные значения
        eigenvals, _ = np.linalg.eigh(L)
        eigenvals = np.sort(eigenvals)

        # Определяем количество кластеров через eigengap
        gaps = [eigenvals[i + 1] - eigenvals[i] for i in range(len(eigenvals) - 1)]
        max_gap_idx = np.argmax(gaps)
        k = max_gap_idx + 1  # количество кластеров

        # Ограничиваем минимальное и максимальное количество кластеров
        k = max(1, min(k, n // 2))
        if self.max_clusters is not None:
            k = min(k, self.max_clusters)

        self.optimal_k = k

        # Спектральная кластеризация
        sc = SpectralClustering(
            n_clusters=k,
            affinity='precomputed',
            random_state=42
        )
        labels = sc.fit_predict(adjacency)

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
            'method': 'spectral',
            'optimal_k': self.optimal_k,
            'max_clusters': self.max_clusters,
            'min_cluster_size': self.min_cluster_size
        }