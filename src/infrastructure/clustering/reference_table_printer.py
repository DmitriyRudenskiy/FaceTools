from typing import List, Any
import os


class ReferenceTablePrinter:
    def __init__(
        self,
        similarity_matrix: List[List[Any]],
        image_paths: List[str],
        num_images: int,
    ):
        self.similarity_matrix = similarity_matrix
        self.image_paths = image_paths
        self.num_images = num_images

    def _print_reference_table(self) -> None:
        """Выводит таблицу сопоставления с эталонами."""
        # Определяем индексы эталонов (файлы с 'refer_' в имени)
        refer_indices = []
        refer_names = []  # Для заголовков таблицы
        # Определяем индексы остальных файлов (без 'refer_' в имени)
        other_indices = []
        other_names = []

        # Собираем индексы и имена
        for i, path in enumerate(self.image_paths):
            filename = os.path.basename(path)
            if "refer_" in filename:
                refer_indices.append(i)
                refer_names.append(filename)
            else:
                other_indices.append(i)
                other_names.append(filename)

        # Если нет эталонов, выходим
        if not refer_indices:
            print(
                "Нет изображений с 'refer_' в имени для использования в качестве эталонов."
            )
            return

        # Определяем ширину столбцов
        max_refer_name_len = (
            max(len(name) for name in refer_names) if refer_names else 0
        )
        max_other_name_len = (
            max(len(name) for name in other_names) if other_names else 0
        )
        cell_width = (
            max(max_refer_name_len, max_other_name_len, 8) + 2
        )  # Минимальная ширина 8 символов

        # Заголовок таблицы
        header = " " * (cell_width + 2)  # Отступ для номеров строк
        for name in refer_names:
            header += f"{name[:cell_width]:<{cell_width}} "
        print(header)
        print("-" * len(header))

        # Строки таблицы
        for j, other_idx in enumerate(other_indices):
            row = f"{other_names[j]:<{cell_width}} |"
            for i, refer_idx in enumerate(refer_indices):
                similarity = self.similarity_matrix[refer_idx][other_idx]
                if similarity is not None:
                    # Форматируем значение: знак + для совпадения, - для несовпадения
                    sign = "+" if similarity[0] else "-"
                    value = f"{sign}{similarity[1]:.2f}"
                    row += f" {value:>{cell_width - 1}}"
                else:
                    row += f" {'-':^{cell_width}}"
            print(row)
