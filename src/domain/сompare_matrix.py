import numpy as np
import json
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path


class CompareMatrix:
    """
    Матрица для хранения результатов сравнения изображений с поддержкой легенды.
    Каждый элемент содержит кортеж (bool, float) или None.
    """

    def __init__(self, legend: List[str]):
        """
        Инициализирует квадратную матрицу на основе легенды.

        Args:
            legend: Список идентификаторов (например, имен файлов изображений).

        Raises:
            ValueError: Если легенда пуста или содержит дубликаты.
        """
        if not legend:
            raise ValueError("Легенда не может быть пустой.")

        if len(legend) != len(set(legend)):
            raise ValueError("Легенда не должна содержать дубликатов.")

        self.size = len(legend)
        self.legend = legend
        self.index_map = {name: idx for idx, name in enumerate(legend)}
        self.matrix = np.full((self.size, self.size), None, dtype=object)

        # Инициализируем диагональ значениями (True, 1.0) - сравнение с самим собой
        for i in range(self.size):
            self.matrix[i, i] = (True, 1.0)

    def get_index(self, name: str) -> int:
        """
        Возвращает индекс элемента по его имени из легенды.

        Args:
            name: Имя элемента из легенды.

        Returns:
            Индекс элемента в матрице.

        Raises:
            KeyError: Если элемент не найден в легенде.
        """
        if name not in self.index_map:
            raise KeyError(f"Элемент '{name}' не найден в легенде.")
        return self.index_map[name]

    def set_value(self, i: int, j: int, value:  float) -> None:
        """
        Устанавливает значение в ячейку (i, j) и симметричную ячейку (j, i).

        Args:
            i: Индекс строки (0 <= i < size).
            j: Индекс столбца (0 <= j < size).
            value: Кортеж (статус: bool, оценка: float) или None.

        Raises:
            IndexError: Если индексы выходят за границы матрицы.
        """
        if not (0 <= i < self.size and 0 <= j < self.size):
            raise IndexError(
                f"Индексы ({i}, {j}) выходят за границы матрицы размером {self.size}x{self.size}."
            )
        self.matrix[i, j] = value
        # Автоматическое обновление симметричной ячейки
        if i != j:
            self.matrix[j, i] = value

    def set_value_by_names(self, name1: str, name2: str, value: Optional[Tuple[bool, float]]) -> None:
        """
        Устанавливает значение по именам элементов из легенды.

        Args:
            name1: Имя первого элемента из легенды.
            name2: Имя второго элемента из легенды.
            value: Кортеж (статус: bool, оценка: float) или None.

        Raises:
            KeyError: Если хотя бы одно имя не найдено в легенде.
        """
        i = self.get_index(name1)
        j = self.get_index(name2)
        self.set_value(i, j, value)

    def get_value(self, i: int, j: int) -> Optional[Tuple[bool, float]]:
        """
        Возвращает значение из ячейки матрицы.

        Args:
            i: Индекс строки (0 <= i < size).
            j: Индекс столбца (0 <= j < size).

        Returns:
            Кортеж (статус, оценка) или None.

        Raises:
            IndexError: Если индексы выходят за границы матрицы.
        """
        if not (0 <= i < self.size and 0 <= j < self.size):
            raise IndexError(
                f"Индексы ({i}, {j}) выходят за границы матрицы размером {self.size}x{self.size}."
            )
        return self.matrix[i, j]

    def get_value_by_names(self, name1: str, name2: str) -> Optional[Tuple[bool, float]]:
        """
        Возвращает значение по именам элементов из легенды.

        Args:
            name1: Имя первого элемента из легенды.
            name2: Имя второго элемента из легенды.

        Returns:
            Кортеж (статус, оценка) или None.

        Raises:
            KeyError: Если хотя бы одно имя не найдено в легенде.
        """
        i = self.get_index(name1)
        j = self.get_index(name2)
        return self.get_value(i, j)

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует матрицу и легенду в словарь для сериализации.

        Returns:
            Словарь с данными матрицы.
        """
        # Преобразуем матрицу в список списков
        matrix_list = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                row.append(self.matrix[i, j])
            matrix_list.append(row)

        return {
            "legend": self.legend,
            "matrix": matrix_list
        }

    def to_json(self, file_path: Union[str, Path]) -> None:
        """
        Сохраняет матрицу и легенду в JSON-файл.

        Args:
            file_path: Путь к файлу для сохранения.
        """
        data = self.to_dict()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompareMatrix':
        """
        Создает объект CompareMatrix из словаря.

        Args:
            data: Словарь с данными матрицы.

        Returns:
            Новый объект CompareMatrix.

        Raises:
            ValueError: Если данные имеют неверный формат.
        """
        if "legend" not in data or "matrix" not in data:
            raise ValueError("Некорректный формат данных: отсутствует легенда или матрица.")

        legend = data["legend"]
        matrix_data = data["matrix"]

        # Создаем объект
        obj = cls(legend)

        # Заполняем матрицу данными
        for i, row in enumerate(matrix_data):
            for j, value in enumerate(row):
                obj.matrix[i, j] = value

        return obj

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> 'CompareMatrix':
        """
        Загружает матрицу и легенду из JSON-файла.

        Args:
            file_path: Путь к файлу для загрузки.

        Returns:
            Новый объект CompareMatrix.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def __repr__(self) -> str:
        """Строковое представление матрицы для отладки."""
        return f"CompareMatrix(legend={self.legend}, size={self.size})"