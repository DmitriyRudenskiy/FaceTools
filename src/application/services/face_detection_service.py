from typing import List, Dict, Any, Optional
import os
import time
import json
import shutil
from src.core.interfaces import (
    FaceDetector,
    BoundingBoxProcessor,
    ImageLoader,
    FileOrganizer,
    ResultSaver,
)
from src.core.exceptions import FaceDetectionError, FileHandlingError, ClusteringError
from src.domain.image_model import Image
from src.domain.face import Face, BoundingBox
from src.infrastructure.comparison.face_recognition import FaceRecognitionFaceComparator
from src.infrastructure.clustering.legacy_image_grouper import ImageGrouper
from src.infrastructure.persistence.group_organizer import GroupOrganizer


class FaceDetectionService:
    """Сервис обработки и кластеризации лиц"""

    def __init__(
        self,
        file_organizer: FileOrganizer,
        face_detector: FaceDetector,
        bbox_processor: BoundingBoxProcessor,
        image_loader: ImageLoader,
        result_saver: ResultSaver,
    ):
        self.file_organizer = file_organizer
        self.face_detector = face_detector
        self.bbox_processor = bbox_processor
        self.image_loader = image_loader
        self.result_saver = result_saver
        self.image_encodings = {}
        self.image_paths = []
        self.similarity_matrix = None
        self.groups_data = []
        self.num_images = 0

    def process(
        self,
        input_path: str,
        output_file: str = "groups.json",
        dest_dir: str = None,
        show_matrix: bool = False,
        show_reference_table: bool = False,
    ) -> bool:
        """Основной метод обработки - совместимый с CLI"""
        return self.process_images(input_path, output_file, dest_dir)

    def process_images(
        self, input_path: str, output_file: str = "groups.json", dest_dir: str = None
    ) -> bool:
        """Обрабатывает изображения и группирует их по схожести лиц."""
        if not os.path.exists(input_path):
            print(f"Ошибка: Директория '{input_path}' не существует")
            return False

        if not os.path.isdir(input_path):
            print(f"Ошибка: '{input_path}' не является директорией")
            return False

        # Если указана директория назначения, проверяем её или создаём
        if dest_dir:
            try:
                os.makedirs(dest_dir, exist_ok=True)
            except OSError as e:
                print(f"Ошибка при работе с директорией назначения '{dest_dir}': {e}")
                return False

        print("=== Начало анализа изображений ===")
        total_start_time = time.time()

        # Загружаем изображения и получаем энкодинги
        self._load_images(input_path)

        # Проверяем, были ли загружены изображения с кодировкой
        num_loaded_images = len(self.image_encodings)
        if num_loaded_images == 0:
            print("Нет изображений с лицами для анализа.")
            # Создаем пустой JSON файл
            empty_result = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "total_groups": 0,
                "unrecognized_count": 0,
                "groups": [],
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(empty_result, f, ensure_ascii=False, indent=4)
            print(f"Создан пустой файл результатов: {output_file}")
            return True

        print(f"Успешно загружено лиц: {num_loaded_images}")

        # Создаем и заполняем матрицу схожести
        self._build_similarity_matrix()

        # Группируем изображения
        grouper = ImageGrouper(self.similarity_matrix, self.image_paths)
        clustering_result = grouper.cluster(self.image_paths)

        # Организуем файлы по группам, если указан путь назначения
        if dest_dir and clustering_result.clusters:
            self._organize_files(clustering_result.clusters, dest_dir)
        elif dest_dir:
            print("Организация файлов не будет выполнена, так как не найдено групп.")

        # Сохраняем результаты в JSON
        self._save_results(
            clustering_result, output_file, total_start_time, num_loaded_images
        )

        return True

    def _load_images(self, directory_path: str):
        """Загружает изображения из директории и получает их энкодинги."""
        print(f"Загружаю изображения из директории: {directory_path}")
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        image_paths = [
            os.path.join(directory_path, filename)
            for filename in os.listdir(directory_path)
            if filename.lower().endswith(valid_extensions)
        ]

        self.image_encodings = {}
        self.image_paths = []
        sequential_index = 0

        for image_path in image_paths:
            try:
                # Загружаем изображение
                image = self.image_loader.load(image_path)

                # Детектируем лица
                boxes = self.face_detector.detect(image)
                if not boxes:
                    print(
                        f"Предупреждение: Не найдено лиц на изображении {os.path.basename(image_path)}"
                    )
                    continue

                # Для каждого лица получаем энкодинг
                face_encodings = []
                for box in boxes:
                    # Обрезаем лицо
                    crop_coords = self.bbox_processor.calculate_square_crop(
                        box, (image.info.size[0], image.info.size[1])
                    )
                    face_image = image.data.crop(crop_coords)

                    # Получаем энкодинг (здесь нужно интегрировать подходящую библиотеку)
                    # Временная заглушка, так как face_recognition не работает с PIL напрямую
                    # В реальной реализации нужно использовать подходящую библиотеку
                    face_encoding = self._get_face_encoding(face_image)
                    if face_encoding is not None:
                        face_encodings.append(face_encoding)

                if face_encodings:
                    # Используем первый энкодинг для упрощения
                    self.image_encodings[sequential_index] = {
                        "path": image_path,
                        "encoding": face_encodings[0],
                    }
                    self.image_paths.append(image_path)
                    sequential_index += 1
                else:
                    print(
                        f"Предупреждение: Не удалось получить энкодинги для {os.path.basename(image_path)}"
                    )

            except Exception as e:
                print(f"Ошибка при обработке {os.path.basename(image_path)}: {e}")

    def _get_face_encoding(self, face_image):
        """Получает энкодинг лица из изображения.
        В реальной реализации здесь будет интеграция с face_recognition или другой библиотекой.
        """
        # Заглушка для демонстрации
        # В реальной реализации нужно использовать подходящую библиотеку
        try:
            import face_recognition
            import numpy as np

            # Конвертируем PIL изображение в numpy array
            face_array = np.array(face_image)
            # Получаем энкодинг
            encodings = face_recognition.face_encodings(face_array)
            if encodings:
                return encodings[0]
            return None
        except ImportError:
            # Если face_recognition не установлен, возвращаем заглушку
            import numpy as np

            return np.random.rand(128)
        except Exception as e:
            print(f"Ошибка при получении энкодинга: {e}")
            return None

    def _build_similarity_matrix(self):
        """Строит матрицу схожести между всеми изображениями."""
        self.num_images = len(self.image_encodings)
        self.similarity_matrix = [
            [None] * self.num_images for _ in range(self.num_images)
        ]

        # Заполняем матрицу
        chunk_size = max(
            1, self.num_images // 4
        )  # Обеспечиваем минимальный размер порции
        print(f"Размер порции для обработки: {chunk_size}")

        for i in range(4):
            start_row = i * chunk_size
            if i == 3:  # Последняя итерация обрабатывает остаток
                end_row = self.num_images
            else:
                end_row = min((i + 1) * chunk_size, self.num_images)

            if start_row >= self.num_images:
                break  # Избегаем лишних итераций, если изображений меньше 4

            print(f"Обрабатываю порцию {i + 1}/4: строки {start_row} до {end_row}")

            for idx1 in range(start_row, end_row):
                for idx2 in range(idx1, self.num_images):
                    if idx1 != idx2:
                        similarity = self._compare_faces(idx1, idx2)
                        self.similarity_matrix[idx1][idx2] = similarity
                        self.similarity_matrix[idx2][idx1] = similarity
                    else:
                        # Диагональ: изображение совпадает с самим собой
                        self.similarity_matrix[idx1][idx2] = [True, 0.0]

    def _compare_faces(self, index1, index2):
        """Сравнивает два лица по индексам."""
        data1 = self.image_encodings.get(index1)
        data2 = self.image_encodings.get(index2)

        if (
            data1 is None
            or data2 is None
            or data1["encoding"] is None
            or data2["encoding"] is None
        ):
            # Если одно из изображений не имеет кодировки, считаем их несовпадающими
            return [False, 1.0]  # Максимальное расстояние

        encoding1 = data1["encoding"]
        encoding2 = data2["encoding"]

        try:
            import face_recognition

            results = face_recognition.compare_faces([encoding1], encoding2)
            face_distance = face_recognition.face_distance([encoding1], encoding2)
            return [results[0], face_distance[0]]
        except ImportError:
            # Если face_recognition не установлен, используем заглушку
            # Здесь можно добавить свою логику сравнения
            import numpy as np

            distance = np.linalg.norm(encoding1 - encoding2)
            return [distance < 0.6, float(distance)]
        except Exception as e:
            print(f"Ошибка при сравнении лиц {index1} и {index2}: {e}")
            return [False, 1.0]

    def _print_similarity_matrix(self):
        """Выводит матрицу схожести в консоль."""
        if not self.similarity_matrix or self.num_images == 0:
            print("Нет изображений для сравнения.")
            return

        element_width = 7
        header_items = [f"{i + 1:5}" for i in range(self.num_images)]
        header = "      " + " ".join(header_items)
        print(header)

        for i in range(self.num_images):
            file_name = os.path.basename(self.image_paths[i])
            row_header = f"{file_name[:15]:15} | "
            row_values = []

            for j in range(self.num_images):
                value = self.similarity_matrix[i][j]
                if i != j and value is not None:
                    formatted_value = f"{'+' if value[0] else '-'}{value[1]:.2f}"
                    row_values.append(f"{formatted_value:>{element_width}}")
                elif i == j:
                    row_values.append(f"{'  *  ':^{element_width}}")
                else:
                    row_values.append(f"{'  -  ':^{element_width}}")

            print(row_header + " ".join(row_values))

    def _print_reference_table(self):
        """Выводит таблицу сопоставления с эталонами."""
        # Определяем индексы эталонов (файлы с 'refer_' в имени)
        refer_indices = []
        refer_names = []  # Для заголовков таблицы
        # Определяем индексы остальных файлов (без 'refer_' в имени)
        non_refer_indices = []
        non_refer_names = []  # Для строк таблицы

        for i in range(self.num_images):
            filename = os.path.basename(self.image_paths[i])
            if filename.startswith("refer_"):
                refer_indices.append(i)
                refer_names.append(filename)
            else:
                non_refer_indices.append(i)
                non_refer_names.append(filename)

        if not refer_indices or not non_refer_indices:
            print("\nТаблица сопоставления с эталонами:")
            print("Нет эталонов или файлов для сопоставления.")
            return

        # Создаем таблицу: строки - non_refer, столбцы - refer + сумма
        table_data = []
        # Для каждой строки (non_refer файла)
        for non_refer_idx in non_refer_indices:
            row_data = []
            row_distances = []  # Список расстояний для расчета суммы
            # Для каждого столбца (refer файла)
            for refer_idx in refer_indices:
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
                (non_refer_names[non_refer_indices.index(non_refer_idx)], row_data)
            )

        # Сортировка по колонке 'Сумма' (по убыванию)
        table_data.sort(key=lambda x: x[1][-1], reverse=True)

        # Вывод таблицы
        print(
            "\nТаблица сопоставления с эталонами (отсортирована по сумме расстояний):"
        )

        # Заголовок таблицы
        name_col_width = (
            max(
                15, max(len(name) for name in non_refer_names + refer_names + ["Сумма"])
            )
            + 2
        )
        value_col_width = 10

        # Формируем строку заголовков
        header_parts = [f"{'Файл':<{name_col_width}}"]
        for refer_name in refer_names:
            header_parts.append(
                f"{refer_name[:value_col_width - 2]:>{value_col_width}}"
            )
        header_parts.append(f"{'Сумма':>{value_col_width}}")
        header_line = "".join(header_parts)
        print(header_line)
        print("-" * len(header_line))

        # Выводим строки таблицы
        for file_name, row_values in table_data:
            row_parts = [f"{file_name[:name_col_width - 2]:<{name_col_width}}"]
            # Выводим значения расстояний до эталонов
            for value in row_values[:-1]:
                row_parts.append(f"{value:.2f}".rjust(value_col_width))
            # Выводим сумму
            row_parts.append(f"{row_values[-1]:.2f}".rjust(value_col_width))

            print("".join(row_parts))

        # Поиск и вывод файлов с минимальной суммой
        if table_data:
            min_sum = min(row_data[-1] for _, row_data in table_data)
            best_matches = [
                file_name
                for file_name, row_data in table_data
                if row_data[-1] == min_sum
            ]

            print("\nЛучшие совпадения (минимальная сумма расстояний до эталонов):")
            for file_name in best_matches:
                print(f"  {file_name} (Сумма: {min_sum:.2f})")

    def _group_images(self):
        """Группирует изображения по схожести лиц."""
        grouper = ImageGrouper(self.similarity_matrix, self.image_paths)
        groups_data = grouper.group_images()

        # Выводим информацию о группах
        print("\nГруппировка завершена:")
        for group_data in groups_data:
            print(
                f"Группа {group_data['id']} (представлена {group_data['representative']}):"
            )
            for path in group_data["images"]:
                print(f"  {path}")
            print()

        print(f"Найдено групп: {len(groups_data)}")
        return groups_data

    # Добавьте в класс FaceDetectionService следующие методы:

    def _organize_files(self, clusters, destination_directory):
        """Организует файлы по группам в отдельные каталоги."""
        print("=== Начало организации файлов по группам ===")
        start_time = time.time()
        total_copied = 0

        for cluster in clusters:
            # Используем имя файла представителя (без расширения) как имя каталога
            representative_name = os.path.splitext(cluster.representative)[0]
            # Очищаем имя от недопустимых символов для имен файлов/каталогов
            safe_group_name = "".join(
                c for c in representative_name if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()
            # Если имя оказалось пустым, используем ID группы
            if not safe_group_name:
                safe_group_name = f"Group_{cluster.id}"
            group_directory_path = os.path.join(destination_directory, safe_group_name)
            print(f"Создаю каталог для группы {cluster.id}: {group_directory_path}")
            # Создаем каталог для группы
            try:
                os.makedirs(group_directory_path, exist_ok=True)
            except OSError as e:
                print(f"Ошибка создания каталога {group_directory_path}: {e}")
                continue  # Пропускаем эту группу, если не удалось создать каталог

            # Копируем файлы группы в созданный каталог
            copied_count = 0
            for full_path in cluster.members_paths:
                try:
                    filename = os.path.basename(full_path)
                    destination_file_path = os.path.join(group_directory_path, filename)
                    # Копируем файл
                    shutil.copy2(full_path, destination_file_path)
                    copied_count += 1
                    total_copied += 1
                except Exception as e:
                    print(
                        f"Ошибка копирования файла {full_path} в {group_directory_path}: {e}"
                    )
            print(f"  Скопировано файлов в группу '{safe_group_name}': {copied_count}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"=== Организация файлов завершена ===")
        print(f"Всего скопировано файлов: {total_copied}")
        print(f"Время на организацию: {elapsed_time:.2f} секунд")

    def _save_results(
        self, clustering_result, output_file, total_start_time, num_loaded_images
    ):
        """Сохраняет результаты в JSON файл."""
        total_elapsed_time = time.time() - total_start_time
        # Формируем итоговый JSON
        result_json = {
            "timestamp": clustering_result.timestamp,
            "total_groups": clustering_result.total_clusters,
            "unrecognized_count": clustering_result.unrecognized_count,
            "groups": [
                {
                    "id": cluster.id,
                    "size": cluster.size,
                    # Опционально: если representative тоже должен быть полным путем
                    # "representative": cluster.representative_path,
                    # Иначе оставьте как есть, если representative - имя файла достаточно
                    "representative": cluster.representative,
                    "representative_full_path": cluster.representative_path,
                    # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
                    # "images": cluster.members, # Было: только имена файлов
                    "images": cluster.members_paths,  # Стало: полные пути
                    # ------------------------
                    "images_full_paths": cluster.members_paths,  # Это дублирует images, если images теперь полные пути
                    "average_similarity": cluster.average_similarity,
                }
                for cluster in clustering_result.clusters
            ],
            "unrecognized_images": clustering_result.unrecognized_images,
        }
        # Сохраняем в файл
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=4)
            print(f"=== Анализ завершен ===")
            print(f"Результаты сохранены в файл: {output_file}")
            print(f"Общее время выполнения: {total_elapsed_time:.2f} секунд")
            print(f"Обработано изображений: {num_loaded_images}")
            print(f"Найдено групп: {clustering_result.total_clusters}")
            print(f"Нераспознанных изображений: {clustering_result.unrecognized_count}")
        except Exception as e:
            print(f"Ошибка при сохранении файла {output_file}: {e}")

        # Сохраняем в файл
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=4)
            print(f"=== Анализ завершен ===")
            print(f"Результаты сохранены в файл: {output_file}")
            print(f"Общее время выполнения: {total_elapsed_time:.2f} секунд")
            print(f"Обработано изображений: {num_loaded_images}")
            print(f"Найдено групп: {clustering_result.total_clusters}")
            print(f"Нераспознанных изображений: {clustering_result.unrecognized_count}")
        except Exception as e:
            print(f"Ошибка при сохранении файла {output_file}: {e}")
