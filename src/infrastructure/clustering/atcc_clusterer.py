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

class ATCCClusterer:
    """Адаптивный пороговый метод (Connected Components with Adaptive Thresholding)"""

    def __init__(self, min_cluster_size: int = 2):
        """
        Args:
            min_cluster_size: минимальный размер кластера
        """
        self.min_cluster_size = min_cluster_size
        self.threshold = None

    def cluster(self, similarity_matrix: List[List[float]]) -> List[List[int]]:
        n = len(similarity_matrix)

        # Собираем все недиагональные элементы матрицы (i < j)
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(similarity_matrix[i][j])

        if not similarities:
            return []  # только одна точка

        # Сортируем схожести и находим наибольший разрыв
        sorted_sims = sorted(similarities)
        if len(sorted_sims) > 1:
            gaps = [sorted_sims[i + 1] - sorted_sims[i] for i in range(len(sorted_sims) - 1)]
            max_gap_index = gaps.index(max(gaps))
            self.threshold = (sorted_sims[max_gap_index] + sorted_sims[max_gap_index + 1]) / 2
        else:
            self.threshold = sorted_sims[0]  # Все значения одинаковы

        # Построение графа на основе порога
        graph = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] >= self.threshold:
                    graph[i].append(j)
                    graph[j].append(i)

        # Поиск связных компонент (DFS)
        visited = [False] * n
        components = []
        for i in range(n):
            if not visited[i]:
                component = []
                stack = [i]
                visited[i] = True
                while stack:
                    node = stack.pop()
                    component.append(node)
                    for neighbor in graph[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            stack.append(neighbor)
                components.append(component)

        # Фильтрация по минимальному размеру кластера
        filtered_components = [comp for comp in components if len(comp) >= self.min_cluster_size]
        return filtered_components

    def get_params(self) -> Dict:
        return {
            'method': 'atcc',
            'threshold': self.threshold,
            'min_cluster_size': self.min_cluster_size
        }