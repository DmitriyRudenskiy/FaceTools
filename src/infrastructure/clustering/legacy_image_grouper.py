# src/infrastructure/clustering/legacy_image_grouper.py
from src.core.interfaces.clusterer import Clusterer
from src.domain.cluster import Cluster, ClusteringResult
import os
import time
from typing import List, Dict, Any


class ImageGrouper(Clusterer):
    """Группировка изображений по схожести лиц"""

    def __init__(self, similarity_matrix, image_paths):
        self.similarity_matrix = similarity_matrix
        self.image_paths = image_paths
        self.num_images = len(image_paths)
        self.groups = []
        self.used_indices = set()  # Отслеживаем, какие изображения уже добавлены в группы

    def calculate_average_distance(self, group_indices):
        """Вычисляет среднее расстояние для каждого изображения в группе."""
        distances = []
        for i in group_indices:
            total_distance = 0.0
            count = 0
            for j in group_indices:
                if i != j and self.similarity_matrix[i][j] is not None:
                    distance = self.similarity_matrix[i][j][1]
                    total_distance += distance
                    count += 1
            if count > 0:
                average_distance = total_distance / count
            else:
                average_distance = float('inf')  # Или 0, если считать, что одиночка близка к себе?
            distances.append((average_distance, i))
        return distances

    def group_images(self):
        """Группирует изображения, обходя матрицу по строкам."""
        start_time = time.time()
        print("Начинаю группировку изображений построчно...")
        self.groups = []  # Очищаем предыдущие группы
        self.used_indices = set()  # Очищаем использованные индексы
        # Проходим по каждой строке (каждому изображению)
        for i in range(self.num_images):
            # Если изображение уже в группе, пропускаем
            if i in self.used_indices:
                continue
            # Начинаем новую группу с текущего изображения
            current_group = [i]
            self.used_indices.add(i)
            # Проверяем все последующие изображения в строке
            for j in range(i + 1, self.num_images):
                # Если изображение j уже использовано, пропускаем
                if j in self.used_indices:
                    continue
                # Получаем результат сравнения из матрицы
                result = self.similarity_matrix[i][j]
                # Проверяем, похожи ли лица (result[0] == True)
                if result is not None and result[0]:
                    # Добавляем изображение j в текущую группу
                    current_group.append(j)
                    self.used_indices.add(j)
                    # Примечание: В оригинальном запросе не указано, нужно ли продолжать
                    # проверку строки после добавления элемента. Здесь мы проверяем
                    # всю строку i.
            # Если группа содержит более одного элемента, сохраняем её
            if len(current_group) > 1:
                self.groups.append(current_group)
            # else:
            #     print(f"Изображение {self.image_paths[i]} не имеет пары.")

        # --- Подготовка данных для возврата (с сортировкой) ---
        # Сначала сортируем сами группы по размеру (количество элементов), от большей к меньшей
        # self.groups - это список списков индексов
        self.groups.sort(key=len, reverse=True)  # Сортировка по длине (размеру группы) по убыванию

        final_groups_data = []
        # Теперь итерируемся по отсортированному списку групп
        for i, group_indices in enumerate(self.groups):
            # Рассчитываем средние расстояния внутри группы
            distances = self.calculate_average_distance(group_indices)
            # Находим изображение с минимальным средним расстоянием (представитель)
            if distances:  # Убедиться, что список не пуст
                min_avg_distance_index = min(distances, key=lambda x: x[0])[1]
            else:
                # Если по какой-то причине расстояний нет, берем первый
                min_avg_distance_index = group_indices[0]
            representative_image_path = self.image_paths[min_avg_distance_index]
            # Подготавливаем данные для JSON
            group_filenames = [os.path.basename(self.image_paths[idx]) for idx in group_indices]
            group_full_paths = [self.image_paths[idx] for idx in group_indices]
            representative_filename = os.path.basename(representative_image_path)
            group_data = Cluster(
                id=i + 1,  # ID теперь соответствует новому порядку
                size=len(group_indices),
                representative=representative_filename,
                representative_path=representative_image_path,
                members=group_filenames,
                members_paths=group_full_paths,
                average_similarity=1.0 - min(distances, key=lambda x: x[0])[0] if distances else 0.0
            )
            final_groups_data.append(group_data)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Группировка завершена за {elapsed_time:.2f} секунд")
        return final_groups_data  # Возвращаем подготовленные и отсортированные данные

    def print_groups(self):
        start_time = time.time()
        # groups_data теперь содержит подготовленные данные для JSON
        groups_data = self.group_images()
        end_time = time.time()
        grouping_time = end_time - start_time
        for group_data in groups_data:
            print(f"Группа {group_data.id} (представлена {group_data.representative}):")
            for path in group_data.members:
                print(f"  {path}")
            print()
        print(f"Общее время группировки: {grouping_time:.2f} секунд")
        print(f"Найдено групп: {len(groups_data)}")
        return groups_data  # Возвращаем данные

    def cluster(self, image_paths: List[str]) -> ClusteringResult:
        """Выполняет кластеризацию и возвращает результат"""
        # В данном случае image_paths уже переданы в конструкторе
        groups_data = self.group_images()

        # Подготавливаем нераспознанные изображения
        all_indices = set(range(self.num_images))
        used_indices_in_groups = set()
        for group in groups_data:
            # Извлекаем индексы из полных путей
            for path in group.members_paths:
                # Находим индекс по пути
                for idx, p in enumerate(self.image_paths):
                    if p == path:
                        used_indices_in_groups.add(idx)
                        break
        unrecognized_indices = all_indices - used_indices_in_groups
        unrecognized_images = []
        for idx in unrecognized_indices:
            full_path = self.image_paths[idx]
            filename = os.path.basename(full_path)
            unrecognized_images.append({
                "filename": filename,
                "full_path": full_path
            })

        return ClusteringResult(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            total_clusters=len(groups_data),
            unrecognized_count=len(unrecognized_images),
            clusters=groups_data,
            unrecognized_images=unrecognized_images
        )