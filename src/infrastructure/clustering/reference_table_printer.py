"""
Модуль для создания таблицы сопоставления с эталонами.
"""

from typing import List, Dict, Any
import os


class ReferenceTablePrinter:
    """Класс для создания и вывода таблицы сопоставления с эталонами."""

    def __init__(self, similarity_matrix, image_paths, num_images):
        self.similarity_matrix = similarity_matrix
        self.image_paths = image_paths
        self.num_images = num_images

        # Определяем индексы эталонов (файлы с 'refer_' в имени)
        self.refer_indices = []
        self.refer_names = []  # Для заголовков таблицы
        # Определяем индексы остальных файлов (без 'refer_' в имени)
        self.non_refer_indices = []
        self.non_refer_names = []  # Для строк таблицы

        for i in range(self.num_images):
            filename = os.path.basename(self.image_paths[i])
            if filename.startswith("refer_"):
                self.refer_indices.append(i)
                self.refer_names.append(filename)
            else:
                self.non_refer_indices.append(i)
                self.non_refer_names.append(filename)

    def print_table(self):
        """Создает и выводит таблицу согласно заданию."""
        if not self.refer_indices or not self.non_refer_indices:
            print("\nТаблица сопоставления с эталонами:")
            print("Нет эталонов или файлов для сопоставления.")
            return

        # Создаем таблицу: строки - non_refer, столбцы - refer + сумма
        table_data = []
        # Для каждой строки (non_refer файла)
        for non_refer_idx in self.non_refer_indices:
            row_data = []
            row_distances = []  # Список расстояний для расчета суммы
            # Для каждого столбца (refer файла)
            for refer_idx in self.refer_indices:
                # Получаем расстояние из матрицы схожести
                similarity_result = self.similarity_matrix[non_refer_idx][refer_idx]
                if similarity_result is not None:
                    distance = similarity_result[1]  # face_distance
                    row_data.append(distance)
                    row_distances.append(distance)
                else:
                    # Если сравнение не удалось, используем максимальное расстояние
                    row_data.append(1.0)
                    row_distances.append(1.0)

            # Вычисляем сумму расстояний для этой строки
            row_sum = sum(row_distances)
            row_data.append(row_sum)

            # Добавляем имя файла строки и данные строки в таблицу
            table_data.append(
                (
                    self.non_refer_names[self.non_refer_indices.index(non_refer_idx)],
                    row_data,
                )
            )

        # --- Сортировка по колонке 'Сумма' (по убыванию) ---
        # Индекс колонки 'Сумма' - это последний элемент в row_data
        table_data.sort(
            key=lambda x: x[1][-1], reverse=True
        )  # reverse=True для сортировки по убыванию

        # --- Вывод таблицы ---
        print(
            "\nТаблица сопоставления с эталонами (отсортирована по сумме расстояний):"
        )

        # Заголовок таблицы
        # Ширина столбца имени файла
        name_col_width = (
            max(
                15,
                max(
                    len(name)
                    for name in self.non_refer_names + self.refer_names + ["Сумма"]
                ),
            )
            + 2
        )
        # Ширина столбцов значений
        value_col_width = 10

        # Формируем строку заголовков
        header_parts = [f"{'Файл':<{name_col_width}}"]
        for refer_name in self.refer_names:
            header_parts.append(
                f"{refer_name[:value_col_width - 2]:>{value_col_width}}"
            )
        header_parts.append(f"{'Сумма':>{value_col_width}}")
        header_line = "".join(header_parts)
        print(header_line)
        print("-" * len(header_line))

        # Выводим строки таблицы
        for file_name, row_values in table_data:  # Используем отсортированные данные
            row_parts = [f"{file_name[:name_col_width - 2]:<{name_col_width}}"]
            # Выводим значения расстояний до эталонов
            for value in row_values[:-1]:  # Все кроме последнего (суммы)
                row_parts.append(f"{value:.2f}".rjust(value_col_width))
            # Выводим сумму
            row_parts.append(f"{row_values[-1]:.2f}".rjust(value_col_width))

            print("".join(row_parts))

        # --- Поиск и вывод файлов с минимальной суммой ---
        if table_data:
            # Находим минимальную сумму
            min_sum = min(row_data[-1] for _, row_data in table_data)
            # Находим все файлы с этой минимальной суммой
            best_matches = [
                file_name
                for file_name, row_data in table_data
                if row_data[-1] == min_sum
            ]

            print("\nЛучшие совпадения (минимальная сумма расстояний до эталонов):")
            for file_name in best_matches:
                print(f"  {file_name} (Сумма: {min_sum:.2f})")
