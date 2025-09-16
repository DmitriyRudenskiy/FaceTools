import json
import os
from typing import Any, List, Optional

import numpy as np

from src.core.interfaces.clusterer import Clusterer
from src.core.interfaces.face_comparator import FaceComparator
from src.domain.cluster import Cluster, ClusteringResult
from src.infrastructure.comparison.deepface_comparator import DeepFaceComparator


# Просто пример для демонстрации
class MockComparator:
    def get_image_path(self, index):
        return f"image_{index}.png"

    def compare(self, i, j):
        # В реальности это может быть расстояние между изображениями i и j
        return np.random.uniform(0, 1)


class MatrixClusterer(Clusterer):
    """
    Реализация интерфейса Clusterer, использующая матрицу сравнения для кластеризации.
    """

    def __init__(self, comparator: FaceComparator, threshold: float = 0.6):
        """
        Инициализирует кластеризатор

        Args:
            comparator: Компаратор для сравнения лиц
            threshold: Пороговое значение для кластеризации
        """
        self.comparator = comparator
        self.threshold = threshold

    def cluster(self, image_paths: List[str]) -> ClusteringResult:
        """Выполняет кластеризацию и возвращает результат"""
        if not image_paths:
            return ClusteringResult(
                timestamp="",
                total_clusters=0,
                unrecognized_count=0,
                clusters=[],
                unrecognized_images=[],
            )

        # Создаем и заполняем матрицу
        matrix = CompareMatrix(len(image_paths))
        matrix.fill(self.comparator, image_paths)

        # Преобразуем в результат кластеризации
        result = matrix.to_clustering_result(self.threshold, image_paths)
        return result

    # В существующий класс CompareMatrix добавляем методы:

    def fill_deepface(
        self, comparator: DeepFaceComparator, image_paths: List[str]
    ) -> None:
        """
        Заполняет матрицу схожести на основе DeepFace компаратора
        Args:
            comparator: Реализация DeepFaceComparator для сравнения лиц
            image_paths: Список путей к изображениям
        """
        size = self.matrix.shape[0]
        self.legend = image_paths[:size]  # Копируем пути к изображениям

        # Инициализируем компаратор
        comparator.init(image_paths)

        for i in range(size):
            for j in range(i, size):  # Заполняем только верхний треугольник
                if i == j:
                    self.matrix[i, j] = np.nan  # Диагональ - NaN
                else:
                    # Получаем расстояние напрямую
                    distance = comparator._compare_by_index(i, j)
                    self.matrix[i, j] = distance
                    self.matrix[j, i] = distance  # Симметричная матрица
