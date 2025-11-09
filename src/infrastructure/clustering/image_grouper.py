from typing import List, Dict
from src.core.interfaces.clusterer import Clusterer
from src.domain.cluster import ClusteringResult


class ImageGrouper():
    """Группировка изображений по схожести лиц с поддержкой различных алгоритмов кластеризации через внедрение зависимостей"""

    def __init__(self, similarity_matrix, image_paths, clusterer: Clusterer):
        """Args:
            similarity_matrix: матрица схожести N x N
            image_paths: пути к изображениям
            clusterer: экземпляр Clusterer, реализующий алгоритм кластеризации
        """
        self.similarity_matrix = similarity_matrix
        self.image_paths = image_paths
        self.num_images = len(image_paths)
        self.clusterer = clusterer
        self.groups = []
        self.evaluation_metrics = {}

        # Валидация матрицы
        self._validate_matrix()

    def _validate_matrix(self):
        """Проверяет корректность матрицы схожести"""
        # Проверка размерности
        if len(self.similarity_matrix) != self.num_images:
            raise ValueError("Размер матрицы схожести не совпадает с количеством изображений")

        # Проверка симметричности и диапазона значений
        for i in range(self.num_images):
            if abs(self.similarity_matrix[i][i] - 1.0) > 1e-5:
                raise ValueError(
                    f"Диагональный элемент [{i},{i}] должен быть равен 1.0, найдено {self.similarity_matrix[i][i]}")

            for j in range(i + 1, self.num_images):
                if abs(self.similarity_matrix[i][j] - self.similarity_matrix[j][i]) > 1e-5:
                    raise ValueError(
                        f"Матрица не симметрична: [{i},{j}]={self.similarity_matrix[i][j]}, [{j},{i}]={self.similarity_matrix[j][i]}")

                if not (0 <= self.similarity_matrix[i][j] <= 1):
                    raise ValueError(
                        f"Значение схожести [{i},{j}] должно быть в диапазоне [0,1], найдено {self.similarity_matrix[i][j]}")

    def _evaluate_clustering(self, groups: List[List[int]]) -> Dict:
        """Оценивает качество кластеризации без ground truth"""
        if not groups or len(groups) < 2:
            return {
                'silhouette_score': -1,
                'avg_cluster_similarity': 0,
                'cluster_sizes': [len(g) for g in groups]
            }

        # Создаем метки кластеров для silhouette_score
        n = len(self.similarity_matrix)
        labels = np.full(n, -1)
        for cluster_id, group in enumerate(groups):
            for idx in group:
                labels[idx] = cluster_id

        # Вычисляем силуэтный коэффициент
        distance_matrix = 1 - np.array(self.similarity_matrix)
        np.fill_diagonal(distance_matrix, 0)

        try:
            silhouette = silhouette_score(distance_matrix, labels, metric='precomputed')
        except:
            silhouette = -1  # Не можем вычислить (например, один кластер)

        # Средняя схожесть внутри кластеров
        def calculate_avg_similarity(cluster_indices):
            total_similarity = 0.0
            count = 0
            for i_idx, i in enumerate(cluster_indices):
                for j in cluster_indices[i_idx + 1:]:
                    total_similarity += self.similarity_matrix[i][j]
                    count += 1
            return total_similarity / count if count > 0 else 0.0

        avg_similarity = np.mean([calculate_avg_similarity(g) for g in groups])

        self.evaluation_metrics = {
            'silhouette_score': silhouette,
            'avg_cluster_similarity': avg_similarity,
            'cluster_sizes': [len(g) for g in groups]
        }
        return self.evaluation_metrics

    def _calculate_average_similarity(self, group_indices):
        """Вычисляет среднюю схожесть внутри группы"""
        total_similarity = 0.0
        count = 0
        for i_idx, i in enumerate(group_indices):
            for j in group_indices[i_idx + 1:]:
                similarity = self.similarity_matrix[i][j]
                total_similarity += similarity
                count += 1
        return total_similarity / count if count > 0 else 0.0

    def group_images(self):
        """Группирует изображения, используя выбранный метод кластеризации."""
        start_time = time.time()
        method_name = self.clusterer.get_params()['method']
        print(f"Начинаю группировку изображений методом '{method_name}'...")

        # Выполняем кластеризацию
        self.groups = self.clusterer.cluster(self.similarity_matrix)

        # Оценка качества кластеризации
        self._evaluate_clustering(self.groups)
        print(f"Качество кластеризации: силуэтный коэффициент = {self.evaluation_metrics['silhouette_score']:.4f}, "
              f"средняя схожесть = {self.evaluation_metrics['avg_cluster_similarity']:.4f}")
        print(f"Размеры кластеров: {self.evaluation_metrics['cluster_sizes']}")

        # Сортируем группы по размеру (от большей к меньшей)
        self.groups.sort(key=len, reverse=True)

        # Подготовка данных для возврата
        final_groups_data = []
        for i, group_indices in enumerate(self.groups):
            # Рассчитываем средние расстояния внутри группы
            avg_similarity = self._calculate_average_similarity(group_indices)

            # Находим изображение с максимальной средней схожестью (представитель)
            max_similarity = -1
            representative_idx = group_indices[0]
            for idx in group_indices:
                total_similarity = 0
                count = 0
                for j in group_indices:
                    if idx != j:
                        total_similarity += self.similarity_matrix[idx][j]
                        count += 1
                if count > 0:
                    avg_idx_similarity = total_similarity / count
                    if avg_idx_similarity > max_similarity:
                        max_similarity = avg_idx_similarity
                        representative_idx = idx

            representative_image_path = self.image_paths[representative_idx]
            group_filenames = [self.image_paths[idx] for idx in group_indices]
            group_full_paths = [self.image_paths[idx] for idx in group_indices]
            representative_filename = os.path.basename(representative_image_path)

            group_data = Cluster(
                id=i + 1,
                size=len(group_indices),
                representative=representative_filename,
                representative_path=representative_image_path,
                members=group_filenames,
                members_paths=group_full_paths,
                average_similarity=avg_similarity,
            )
            final_groups_data.append(group_data)

        end_time = time.time()
        print(f"Группировка завершена за {end_time - start_time:.2f} секунд. "
              f"Найдено {len(final_groups_data)} групп.")

        return final_groups_data

    def print_groups(self):
        start_time = time.time()
        groups_data = self.group_images()
        end_time = time.time()
        grouping_time = end_time - start_time
        for group_data in groups_data:
            print(f"Группа {group_data.id} (представлена {group_data.representative}, "
                  f"схожесть={group_data.average_similarity:.4f}):")
            for path in group_data.members:
                print(f"  {path}")
            print()
        print(f"Общее время группировки: {grouping_time:.2f} секунд")
        print(f"Найдено групп: {len(groups_data)}")
        return groups_data

    def cluster(self) -> ClusteringResult:
        """Выполняет кластеризацию и возвращает результат"""
        groups_data = self.group_images()

        # Подготавливаем нераспознанные изображения
        all_indices = set(range(self.num_images))
        used_indices_in_groups = set()
        for group in groups_data:
            for path in group.members_paths:
                try:
                    idx = self.image_paths.index(path)
                    used_indices_in_groups.add(idx)
                except ValueError:
                    continue

        unrecognized_indices = all_indices - used_indices_in_groups
        unrecognized_images = [{"filename": self.image_paths[idx], "full_path": self.image_paths[idx]}
                               for idx in unrecognized_indices]

        return ClusteringResult(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            total_clusters=len(groups_data),
            unrecognized_count=len(unrecognized_images),
            clusters=groups_data,
            unrecognized_images=unrecognized_images,
            clustering_params=self.clusterer.get_params(),
            evaluation_metrics=self.evaluation_metrics
        )