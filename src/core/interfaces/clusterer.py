from abc import ABC, abstractmethod
from typing import List
from src.domain.cluster import ClusteringResult


class Clusterer(ABC):
    """Абстракция для кластеризации лиц"""

    @abstractmethod
    def cluster(self, image_paths: List[str]) -> ClusteringResult:
        """Выполняет кластеризацию и возвращает результат"""
        pass
