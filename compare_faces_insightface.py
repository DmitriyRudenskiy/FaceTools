import os
import time
import argparse
import sys
import json
import shutil
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo


class FaceComparator:
    def __init__(self, det_size=(640, 640), ctx_id=0, det_thresh=0.5, rec_model='arcface_r100_v1'):
        """
        Инициализация компаратора с использованием InsightFace

        Args:
            det_size: Размер для детекции лиц
            ctx_id: ID устройства (0 - GPU, -1 - CPU)
            det_thresh: Порог уверенности детектора
            rec_model: Модель для распознавания (arcface_r100_v1, arcface_r50_v1 и др.)
        """
        print(f"Инициализация InsightFace (ctx_id={ctx_id}, det_size={det_size})...")
        self.app = FaceAnalysis(
            name=rec_model,
            root='~/.insightface',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if ctx_id >= 0 else ['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
        print(f"Используется модель распознавания: {rec_model}")

        # Порог для сравнения (косинусное расстояние)
        # В InsightFace обычно используется 0.6 для косинусного расстояния
        self.comparison_threshold = 0.6
        self.image_encodings = {}

    def load(self, directory_path):
        """Загрузка и обработка изображений из директории"""
        print(f"Загружаю изображения из директории: {directory_path}")
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = [
            os.path.join(directory_path, filename)
            for filename in os.listdir(directory_path)
            if filename.lower().endswith(valid_extensions)
        ]

        self.image_encodings = {}
        sequential_index = 0
        total_start = time.time()

        for idx, image_path in enumerate(image_paths):
            try:
                # Загрузка изображения с помощью OpenCV
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Предупреждение: Не удалось загрузить изображение {os.path.basename(image_path)}")
                    continue

                # Обнаружение лиц и получение эмбеддингов
                start_time = time.time()
                faces = self.app.get(img)
                processing_time = time.time() - start_time

                if len(faces) > 0:
                    # Используем лицо с наибольшей уверенностью
                    main_face = max(faces, key=lambda x: x.det_score)

                    self.image_encodings[sequential_index] = {
                        'path': image_path,
                        'encoding': main_face.embedding,
                        'det_score': main_face.det_score,
                        'bbox': main_face.bbox.tolist(),
                        'processing_time': processing_time
                    }

                    sequential_index += 1
                    print(f"[{idx + 1}/{len(image_paths)}] Обработано: {os.path.basename(image_path)} "
                          f"(лиц: {len(faces)}, уверенность: {main_face.det_score:.4f}, "
                          f"время: {processing_time:.2f}с)")
                else:
                    print(
                        f"[{idx + 1}/{len(image_paths)}] Предупреждение: Не найдено лиц на изображении {os.path.basename(image_path)}")

            except Exception as e:
                print(f"[{idx + 1}/{len(image_paths)}] Ошибка при обработке {os.path.basename(image_path)}: {str(e)}")

        total_time = time.time() - total_start
        print(f"Загружено лиц: {len(self.image_encodings)} из {len(image_paths)} изображений "
              f"(общее время: {total_time:.2f}с)")

    def compare_two_faces(self, index1, index2):
        """Сравнение двух лиц с использованием косинусного расстояния"""
        data1 = self.image_encodings.get(index1)
        data2 = self.image_encodings.get(index2)

        if data1 is None or data2 is None or data1['encoding'] is None or data2['encoding'] is None:
            return [False, 1.0]  # Максимальное расстояние

        # Вычисление косинусного расстояния
        embedding1 = data1['encoding']
        embedding2 = data2['encoding']

        # Нормализация векторов
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        # Косинусное расстояние = 1 - косинусная симметрия
        cosine_similarity = np.dot(embedding1, embedding2)
        cosine_distance = 1 - cosine_similarity

        # Сравнение с порогом
        is_match = cosine_distance < self.comparison_threshold

        return [is_match, cosine_distance]


class SimilarityMatrix:
    def __init__(self, directory_path, det_size=(640, 640), ctx_id=0, det_thresh=0.5, rec_model='arcface_r100_v1'):
        self.face_comparator = FaceComparator(
            det_size=det_size,
            ctx_id=ctx_id,
            det_thresh=det_thresh,
            rec_model=rec_model
        )
        self.face_comparator.load(directory_path)


class MatrixPrinter:
    def __init__(self, similarity_matrix_instance):
        self.num_images = len(similarity_matrix_instance.face_comparator.image_encodings)
        self.similarity_matrix = [[None] * self.num_images for _ in range(self.num_images)]
        self.face_comparator = similarity_matrix_instance.face_comparator
        self.image_paths = [self.face_comparator.image_encodings[i]['path'] for i in range(self.num_images)]

    def fill(self, start_row=0, end_row=None):
        """Заполнение матрицы схожести с использованием косинусного расстояния"""
        if end_row is None:
            end_row = self.num_images
        start_row = max(0, start_row)
        end_row = min(self.num_images, end_row)

        print(f"Заполнение матрицы схожести (строки {start_row} до {end_row})...")
        start_time = time.time()

        for i in range(start_row, end_row):
            for j in range(i, self.num_images):
                if i != j:
                    similarity = self.face_comparator.compare_two_faces(i, j)
                    # Заполняем симметрично
                    self.similarity_matrix[i][j] = similarity
                    self.similarity_matrix[j][i] = similarity
                else:
                    # Диагональ: изображение совпадает с самим собой
                    self.similarity_matrix[i][j] = [True, 0.0]

            # Прогресс
            if (i + 1) % 5 == 0 or i == end_row - 1:
                elapsed = time.time() - start_time
                progress = (i - start_row + 1) / (end_row - start_row)
                eta = elapsed / progress * (1 - progress) if progress > 0 else 0
                print(f"  Прогресс: {i + 1}/{self.num_images} | "
                      f"Время: {elapsed:.1f}с | "
                      f"Осталось: {eta:.1f}с")

    def get_matrix(self):
        return self.similarity_matrix

    def print_matrix(self):
        """Печать матрицы схожести (основной функционал не меняется)"""
        if not self.similarity_matrix or self.num_images == 0:
            print("Нет изображений для сравнения.")
            return

        element_width = 7
        header_items = [f"{i + 1:5}" for i in range(self.num_images)]
        header = "      " + " ".join(header_items)
        print(header)

        for i, row in enumerate(self.similarity_matrix):
            file_name = os.path.basename(self.image_paths[i])
            row_header = f"{file_name[:15]:15} | "
            row_values = []

            for j, value in enumerate(row):
                if i != j and value is not None:
                    formatted_value = f"{'+' if value[0] else '-'}{value[1]:.2f}"
                    row_values.append(f"{formatted_value:>{element_width}}")
                elif i == j:
                    row_values.append(f"{'  *  ':^{element_width}}")
                else:
                    row_values.append(f"{'  -  ':^{element_width}}")

            print(row_header + " ".join(row_values))


class ImageGrouper:
    def __init__(self, similarity_matrix, image_paths):
        self.similarity_matrix = similarity_matrix
        self.image_paths = image_paths
        self.num_images = len(image_paths)
        self.groups = []
        self.used_indices = set()

    def calculate_average_distance(self, group_indices):
        """Вычисление среднего расстояния для каждого изображения в группе"""
        distances = []
        for i in group_indices:
            total_distance = 0.0
            count = 0
            for j in group_indices:
                if i != j and self.similarity_matrix[i][j] is not None:
                    total_distance += self.similarity_matrix[i][j][1]
                    count += 1
            average_distance = total_distance / count if count > 0 else float('inf')
            distances.append((average_distance, i))
        return distances

    def group_images(self):
        """Группировка изображений по схожести"""
        start_time = time.time()
        print("Начинаю группировку изображений построчно...")
        self.groups = []
        self.used_indices = set()

        for i in range(self.num_images):
            if i in self.used_indices:
                continue

            current_group = [i]
            self.used_indices.add(i)

            for j in range(i + 1, self.num_images):
                if j in self.used_indices:
                    continue

                result = self.similarity_matrix[i][j]
                if result is not None and result[0]:  # result[0] - флаг совпадения
                    current_group.append(j)
                    self.used_indices.add(j)

            if len(current_group) > 1:
                self.groups.append(current_group)

        # Сортировка групп по размеру
        self.groups.sort(key=len, reverse=True)

        # Подготовка данных для вывода
        final_groups_data = []
        for i, group_indices in enumerate(self.groups):
            distances = self.calculate_average_distance(group_indices)
            min_avg_distance_index = min(distances, key=lambda x: x[0])[1] if distances else group_indices[0]

            representative_image_path = self.image_paths[min_avg_distance_index]
            group_filenames = [os.path.basename(self.image_paths[idx]) for idx in group_indices]
            group_full_paths = [self.image_paths[idx] for idx in group_indices]
            representative_filename = os.path.basename(representative_image_path)

            group_data = {
                "id": i + 1,
                "size": len(group_indices),
                "representative": representative_filename,
                "representative_full_path": representative_image_path,
                "images": group_filenames,
                "images_full_paths": group_full_paths
            }
            final_groups_data.append(group_data)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Группировка завершена за {elapsed_time:.2f} секунд")
        return final_groups_data

    def print_groups(self):
        """Вывод групп на экран и возврат данных"""
        start_time = time.time()
        groups_data = self.group_images()
        end_time = time.time()

        for group_data in groups_data:
            print(f"Группа {group_data['id']} (представлена {group_data['representative']}):")
            for path in group_data['images']:
                print(f"  {path}")
            print()

        grouping_time = end_time - start_time
        print(f"Общее время группировки: {grouping_time:.2f} секунд")
        print(f"Найдено групп: {len(groups_data)}")
        return groups_data


class GroupOrganizer:
    """Класс для организации файлов по группам в отдельные каталоги"""

    def __init__(self, groups_data, destination_directory):
        self.groups_data = groups_data
        self.destination_directory = destination_directory
        os.makedirs(self.destination_directory, exist_ok=True)

    def organize(self):
        """Организация файлов по группам"""
        print("=== Начало организации файлов по группам ===")
        start_time = time.time()
        total_copied = 0

        for group_data in self.groups_data:
            group_id = group_data['id']
            representative_name = os.path.splitext(group_data['representative'])[0]
            safe_group_name = "".join(c for c in representative_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            if not safe_group_name:
                safe_group_name = f"Group_{group_id}"

            group_directory_path = os.path.join(self.destination_directory, safe_group_name)
            print(f"Создаю каталог для группы {group_id}: {group_directory_path}")

            try:
                os.makedirs(group_directory_path, exist_ok=True)
            except OSError as e:
                print(f"Ошибка создания каталога {group_directory_path}: {e}")
                continue

            copied_count = 0
            for full_path in group_data['images_full_paths']:
                try:
                    filename = os.path.basename(full_path)
                    destination_file_path = os.path.join(group_directory_path, filename)
                    shutil.copy2(full_path, destination_file_path)
                    copied_count += 1
                    total_copied += 1
                except Exception as e:
                    print(f"Ошибка копирования файла {full_path} в {group_directory_path}: {e}")

            print(f"  Скопировано файлов в группу '{safe_group_name}': {copied_count}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"=== Организация файлов завершена ===")
        print(f"Всего скопировано файлов: {total_copied}")
        print(f"Время на организацию: {elapsed_time:.2f} секунд")


def main():
    parser = argparse.ArgumentParser(
        description='Анализ и группировка изображений по схожести лиц с использованием InsightFace')
    parser.add_argument('-s', '--src', required=True, help='Путь к директории с изображениями для анализа')
    parser.add_argument('-o', '--output', default='groups.json',
                        help='Путь к выходному JSON файлу (по умолчанию groups.json)')
    parser.add_argument('-d', '--dest', help='Путь к директории для создания подкаталогов с группами')

    # Добавляем параметры для InsightFace
    parser.add_argument('--det-size', type=str, default='640,640', help='Размер для детекции лиц (ширина,высота)')
    parser.add_argument('--ctx-id', type=int, default=0, help='ID устройства (0 - GPU, -1 - CPU)')
    parser.add_argument('--det-thresh', type=float, default=0.5, help='Порог уверенности детектора')
    parser.add_argument('--rec-model', type=str, default='arcface_r100_v1',
                        help='Модель распознавания (arcface_r100_v1, arcface_r50_v1 и др.)')

    args = parser.parse_args()

    # Парсим det-size
    try:
        det_width, det_height = map(int, args.det_size.split(','))
        det_size = (det_width, det_height)
    except:
        print("Ошибка: Неверный формат det-size. Используйте 'ширина,высота' (например, '640,640')")
        return

    if not os.path.exists(args.src):
        print(f"Ошибка: Директория '{args.src}' не существует")
        return

    if not os.path.isdir(args.src):
        print(f"Ошибка: '{args.src}' не является директорией")
        return

    if args.dest:
        try:
            os.makedirs(args.dest, exist_ok=True)
        except OSError as e:
            print(f"Ошибка при работе с директорией назначения '{args.dest}': {e}")
            return

    print("=== Начало анализа изображений с использованием InsightFace ===")
    print(f"Параметры модели: det_size={det_size}, ctx_id={args.ctx_id}, "
          f"det_thresh={args.det_thresh}, rec_model={args.rec_model}")

    total_start_time = time.time()

    # Используем InsightFace
    similarity_matrix_instance = SimilarityMatrix(
        args.src,
        det_size=det_size,
        ctx_id=args.ctx_id,
        det_thresh=args.det_thresh,
        rec_model=args.rec_model
    )

    num_loaded_images = len(similarity_matrix_instance.face_comparator.image_encodings)
    if num_loaded_images == 0:
        print("Нет изображений с лицами для анализа.")
        empty_result = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_groups": 0,
            "unrecognized_count": 0,
            "groups": [],
            "unrecognized_images": []
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(empty_result, f, ensure_ascii=False, indent=4)
        print(f"Создан пустой файл результатов: {args.output}")
        sys.exit(0)
    else:
        print(f"Успешно загружено лиц: {num_loaded_images}")

    matrix_printer = MatrixPrinter(similarity_matrix_instance)
    num_encodings = matrix_printer.num_images

    # Обработка матрицы частями
    chunk_size = max(1, num_encodings // 4)
    print(f"Размер порции для обработки: {chunk_size}")

    for i in range(4):
        start_row = i * chunk_size
        end_row = num_encodings if i == 3 else min((i + 1) * chunk_size, num_encodings)

        if start_row >= num_encodings:
            break

        print(f"Обрабатываю порцию {i + 1}/4: строки {start_row} до {end_row}")
        matrix_printer.fill(start_row, end_row)

    # Группировка изображений
    grouper = ImageGrouper(matrix_printer.get_matrix(), matrix_printer.image_paths)
    groups_data = grouper.print_groups()

    # Организация файлов по группам
    if args.dest and groups_data:
        organizer = GroupOrganizer(groups_data, args.dest)
        organizer.organize()
    elif args.dest:
        print("Организация файлов не будет выполнена, так как не найдено групп.")

    # Подготовка данных для JSON
    total_elapsed_time = time.time() - total_start_time

    # Определение нераспознанных изображений
    all_indices = set(range(num_loaded_images))
    used_indices_in_groups = set()

    for group_data in groups_data:
        for path in group_data['images_full_paths']:
            for idx, p in enumerate(matrix_printer.image_paths):
                if p == path:
                    used_indices_in_groups.add(idx)
                    break

    unrecognized_indices = all_indices - used_indices_in_groups
    unrecognized_data = []

    for idx in unrecognized_indices:
        full_path = matrix_printer.image_paths[idx]
        filename = os.path.basename(full_path)
        unrecognized_data.append({
            "filename": filename,
            "full_path": full_path
        })

    # Формирование итогового JSON
    result_json = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_groups": len(groups_data),
        "unrecognized_count": len(unrecognized_data),
        "groups": groups_data,
        "unrecognized_images": unrecognized_data,
        "parameters": {
            "det_size": det_size,
            "ctx_id": args.ctx_id,
            "det_thresh": args.det_thresh,
            "rec_model": args.rec_model,
            "comparison_threshold": 0.6  # Косинусное расстояние
        }
    }

    # Сохранение результатов
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=4)
        print(f"=== Анализ завершен ===")
        print(f"Результаты сохранены в файл: {args.output}")
        print(f"Общее время выполнения: {total_elapsed_time:.2f} секунд")
        print(f"Обработано изображений: {num_loaded_images}")
        print(f"Найдено групп: {len(groups_data)}")
        print(f"Нераспознанных изображений: {len(unrecognized_data)}")
        if args.dest:
            print(f"Файлы организованы в директории: {args.dest}")
    except Exception as e:
        print(f"Ошибка при сохранении файла {args.output}: {e}")


if __name__ == "__main__":
    main()