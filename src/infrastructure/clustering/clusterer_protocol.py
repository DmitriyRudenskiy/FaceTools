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


class Clusterer(Protocol):
    """Интерфейс для всех алгоритмов кластеризации"""

    @abstractmethod
    def cluster(self, similarity_matrix: List[List[float]]) -> List[List[int]]:
        """Выполняет кластеризацию и возвращает список кластеров (список списков индексов)"""
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        """Возвращает параметры кластеризатора"""
        pass