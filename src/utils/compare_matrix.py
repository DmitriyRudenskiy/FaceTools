import numpy as np
import os
import json

class CompareMatrix:
    def __init__(self, size):
        self.matrix = np.zeros((size, size))
        self.legend = []

    def fill(self, comparator):
        """Заполнение матрицы: диагональ - NULL, заполняются только элементы i > j"""
        size = self.matrix.shape[0]

        # Заполняем legend путями к файлам
        self.legend = [comparator.get_image_path(i) for i in range(size)]

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

    def save(self, filepath):
        """
        Сохраняет матрицу в JSON файл

        Args:
            filepath (str): Путь к файлу для сохранения
        """
        try:
            # Создаем директорию, если она не существует
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

            print(f"Матрица успешно сохранена в JSON {filepath}")
        except Exception as e:
            print(f"Ошибка при сохранении матрицы в JSON: {e}")

    def load(self, filepath):
        """
        Загружает матрицу из JSON файла

        Args:
            filepath (str): Путь к файлу для загрузки
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
            print(f"Матрица успешно загружена из JSON {filepath}")
        except Exception as e:
            print(f"Ошибка при загрузке матрицы из JSON: {e}")