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