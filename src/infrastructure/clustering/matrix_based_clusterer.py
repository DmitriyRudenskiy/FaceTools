"""
Модуль для кластеризации лиц на основе матрицы сравнения.
Содержит реализацию CompareMatrix - инструмента для работы с матрицей схожести лиц.
"""

import numpy as np
import os
import json
from typing import List
from src.domain.cluster import Cluster, ClusteringResult
from src.core.interfaces.clusterer import Clusterer
from src.core.interfaces.face_comparator import FaceComparator

# Просто пример для демонстрации
class MockComparator:
    def get_image_path(self, index):
        return f"image_{index}.png"

    def compare(self, i, j):
        # В реальности это может быть расстояние между изображениями i и j
        return np.random.uniform(0, 1)

class CompareMatrix:
    """
    Класс для работы с матрицей сравнения лиц.
    Реализует заполнение, анализ и сохранение матрицы схожести.
    """

    def __init__(self, size: int):
        """Инициализирует матрицу сравнения заданного размера"""
        self.matrix = np.zeros((size, size))
        self.legend = []  # Список путей к изображениям

    def fill(self, comparator: FaceComparator, image_paths: List[str]) -> None:
        """
        Заполняет матрицу схожести на основе компаратора

        Args:
            comparator: Реализация FaceComparator для сравнения лиц
            image_paths: Список путей к изображениям
        """
        size = self.matrix.shape[0]
        self.legend = image_paths[:size]  # Копируем пути к изображениям

        for i in range(size):
            for j in range(i, size):  # Заполняем только верхний треугольник
                if i == j:
                    self.matrix[i, j] = np.nan  # Диагональ - NaN
                else:
                    # Получаем результат сравнения (совпадение, расстояние)
                    _, distance = comparator.compare(image_paths[i], image_paths[j])
                    self.matrix[i, j] = distance
                    self.matrix[j, i] = distance  # Симметричная матрица

    def display(self) -> None:
        """Выводит матрицу на экран в компактном виде"""
        print("Матрица схожести:")
        with np.printoptions(precision=2, suppress=True):
            print(self.matrix)

    def display_full(self) -> None:
        """Выводит матрицу на экран целиком с подробным форматированием"""
        print("Матрица схожести (полный вывод):")
        with np.printoptions(
            threshold=np.inf,
            linewidth=1000,
            precision=2,
            suppress=True,
            formatter={'float_kind': lambda x: f'{x:1.2f}'}
        ):
            print(self.matrix)

    def split(self, threshold: float) -> List[List[int]]:
        """
        Разбивает матрицу на группы по пороговому значению

        Args:
            threshold: Пороговое значение для определения схожести

        Returns:
            Список групп, где каждая группа - список индексов связанных изображений
        """
        size = self.matrix.shape[0]
        if size == 0:
            return []

        # Создаем бинарную матрицу: True если расстояние <= threshold
        mask = np.ma.masked_invalid(self.matrix) <= threshold
        adjacency_matrix = mask.filled(False)
        np.fill_diagonal(adjacency_matrix, False)  # Убираем диагональ

        # Группировка связанных компонентов
        groups = []
        visited = set()

        for i in range(size):
            if i not in visited:
                current_group = []
                stack = [i]

                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        current_group.append(node)

                        # Находим всех соседей
                        neighbors = np.where(adjacency_matrix[node])[0]
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                stack.append(neighbor)

                if current_group:
                    current_group.sort()
                    groups.append(current_group)

        return groups

    def to_clustering_result(self, threshold: float, image_paths: List[str]) -> ClusteringResult:
        """
        Преобразует матрицу в результат кластеризации

        Args:
            threshold: Пороговое значение для кластеризации
            image_paths: Список путей к изображениям

        Returns:
            Объект ClusteringResult с результатами кластеризации
        """
        groups = self.split(threshold)
        clusters = []
        unrecognized_indices = set(range(len(image_paths))) - set(idx for group in groups for idx in group)

        # Создаем кластеры
        for i, group_indices in enumerate(groups):
            # Вычисляем среднее расстояние в группе
            total_distance = 0.0
            count = 0
            for idx in group_indices:
                for other_idx in group_indices:
                    if idx != other_idx and not np.isnan(self.matrix[idx, other_idx]):
                        total_distance += self.matrix[idx, other_idx]
                        count += 1

            avg_similarity = 1.0 - (total_distance / count if count > 0 else 0.0)

            # Находим представителя группы (с минимальным средним расстоянием)
            min_distance = float('inf')
            representative_idx = group_indices[0]
            for idx in group_indices:
                group_distance = sum(self.matrix[idx, other_idx] for other_idx in group_indices if idx != other_idx and not np.isnan(self.matrix[idx, other_idx]))
                if group_distance < min_distance:
                    min_distance = group_distance
                    representative_idx = idx

            # Создаем объект кластера
            representative_path = image_paths[representative_idx]
            clusters.append(Cluster(
                id=i + 1,
                size=len(group_indices),
                representative=os.path.basename(representative_path),
                representative_path=representative_path,
                members=[os.path.basename(image_paths[idx]) for idx in group_indices],
                members_paths=[image_paths[idx] for idx in group_indices],
                average_similarity=avg_similarity
            ))

        # Определяем нераспознанные изображения
        unrecognized_images = [
            {"filename": os.path.basename(image_paths[idx]), "full_path": image_paths[idx]}
            for idx in unrecognized_indices
        ]

        return ClusteringResult(
            timestamp="",
            total_clusters=len(clusters),
            unrecognized_count=len(unrecognized_images),
            clusters=clusters,
            unrecognized_images=unrecognized_images
        )

    def save(self, filepath: str) -> None:
        """
        Сохраняет матрицу в JSON файл

        Args:
            filepath: Путь к файлу для сохранения
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Преобразуем numpy массив в список для JSON сериализации
            matrix_list = self.matrix.tolist()

            # Заменяем NaN значения на null для JSON
            for i in range(len(matrix_list)):
                for j in range(len(matrix_list[i])):
                    if isinstance(matrix_list[i][j], float) and np.isnan(matrix_list[i][j]):
                        matrix_list[i][j] = None

            data = {
                'matrix': matrix_list,
                'legend': self.legend,
                'size': self.matrix.shape[0]
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"Матрица успешно сохранена в {filepath}")
        except Exception as e:
            print(f"Ошибка при сохранении матрицы: {e}")

    def load(self, filepath: str) -> None:
        """
        Загружает матрицу из JSON файла

        Args:
            filepath: Путь к файлу для загрузки
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Файл не найден: {filepath}")

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Восстанавливаем матрицу из списка
            matrix_list = data['matrix']

            # Заменяем null значения обратно на NaN
            for i in range(len(matrix_list)):
                for j in range(len(matrix_list[i])):
                    if matrix_list[i][j] is None:
                        matrix_list[i][j] = np.nan

            self.matrix = np.array(matrix_list)
            self.legend = data.get('legend', [])
            print(f"Матрица успешно загружена из {filepath}")
        except Exception as e:
            print(f"Ошибка при загрузке матрицы: {e}")


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
                unrecognized_images=[]
            )

        # Создаем и заполняем матрицу
        matrix = CompareMatrix(len(image_paths))
        matrix.fill(self.comparator, image_paths)

        # Преобразуем в результат кластеризации
        result = matrix.to_clustering_result(self.threshold, image_paths)
        return result