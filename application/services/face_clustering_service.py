from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from core.interfaces import ClusterAnalyzer, Clusterer
from domain.face import Face


class SilhouetteClusterAnalyzer(ClusterAnalyzer):
    """Реализация анализа кластеров методом силуэта."""

    def find_optimal_clusters(self, embeddings: List[np.ndarray], max_clusters: int, method: str = 'silhouette') -> int:
        embeddings_array = np.array(embeddings)
        best_score = -1
        optimal_k = 2  # Минимальное количество кластеров

        for k in range(2, min(max_clusters + 1, len(embeddings_array))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_array)
            score = silhouette_score(embeddings_array, labels)

            if score > best_score:
                best_score = score
                optimal_k = k

        return optimal_k


class KMeansClusterer(Clusterer):
    """Реализация кластеризации с использованием KMeans."""

    def cluster(self, embeddings: List[np.ndarray], n_clusters: int) -> List[int]:
        embeddings_array = np.array(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(embeddings_array).tolist()


class FaceClusteringService:
    """Сервис для кластеризации лиц."""

    def __init__(self,
                 cluster_analyzer: ClusterAnalyzer,
                 clusterer: Clusterer):
        self.cluster_analyzer = cluster_analyzer
        self.clusterer = clusterer

    def cluster_faces(self,
                      faces: List[Face],
                      max_clusters: int = 20,
                      method: str = 'silhouette') -> Tuple[int, List[int]]:
        """Выполняет кластеризацию лиц."""
        # Извлекаем эмбеддинги
        embeddings = [face.embedding for face in faces]

        if len(embeddings) < 2:
            raise ValueError("Недостаточно лиц для кластеризации (минимум 2)")

        # Поиск оптимального количества кластеров
        optimal_k = self.cluster_analyzer.find_optimal_clusters(embeddings, max_clusters, method)

        # Кластеризация
        labels = self.clusterer.cluster(embeddings, optimal_k)

        return optimal_k, labels