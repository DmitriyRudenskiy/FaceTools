import numpy as np

class CompareMatrix:
    def __init__(self, size):
        self.matrix = np.zeros((size, size))

    def fill(self, comparator):
        """Заполнение матрицы: диагональ - NULL, заполняются только элементы i > j"""
        size = self.matrix.shape[0]
        for i in range(size):
            for j in range(size):
                if i == j:
                    self.matrix[i, j] = np.nan
                elif i > j:
                    self.matrix[i, j] = comparator.compare(i, j)
                    self.matrix[j, i] = self.matrix[i, j]

    def display(self):
        """Вывод матрицы на экран"""
        print("Текущая матрица:")
        print(self.matrix)

    def split(self, threshold):
        """
        Разбивает матрицу на несколько подматриц по пороговому значению

        Args:
            threshold (float): Пороговое значение для разделения

        Returns:
            list: Массив подматриц, где каждая подматрица содержит индексы связанных элементов
        """
        size = self.matrix.shape[0]
        if size == 0:
            return []

        # Создаем бинарную матрицу: True если расстояние <= threshold
        # Используем numpy.ma.masked_invalid для обработки NaN значений
        mask = np.ma.masked_invalid(self.matrix) <= threshold

        # Заменяем masked значения на False
        adjacency_matrix = mask.filled(False)

        # Убираем NaN с диагонали (делаем диагональ False)
        np.fill_diagonal(adjacency_matrix, False)

        # Группировка связанных компонентов
        groups = []
        visited = set()

        for i in range(size):
            if i not in visited:
                # Начинаем новую группу
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
                    # Сортируем группу по возрастанию индексов
                    current_group.sort()
                    groups.append(current_group)

        return groups

    def split_to_matrices(self, threshold):
        """
        Разбивает матрицу на массив подматриц по пороговому значению

        Args:
            threshold (float): Пороговое значение для разделения

        Returns:
            list: Массив подматриц
        """
        groups = self.split(threshold)
        submatrices = []

        for group in groups:
            if len(group) > 0:
                submatrix = self.matrix[np.ix_(group, group)]
                submatrices.append({
                    'indices': group,
                    'matrix': submatrix
                })

        return submatrices