import os
import time
import argparse
import face_recognition
import sys
import json
import shutil  # Импортируем shutil для копирования файлов


class FaceComparator:
    def __init__(self):
        self.image_encodings = {}

    def load(self, directory_path):
        print(f"Загружаю изображения из директории: {directory_path}")
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = [
            os.path.join(directory_path, filename)
            for filename in os.listdir(directory_path)
            if filename.lower().endswith(valid_extensions)
        ]
        self.image_encodings = {}
        sequential_index = 0
        for image_path in image_paths:
            try:
                # Сохраняем полный путь к изображению
                self.image_encodings[sequential_index] = {
                    'path': image_path,
                    'encoding': None
                }
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                if len(face_encodings) > 0 and face_encodings[0] is not None:
                    self.image_encodings[sequential_index]['encoding'] = face_encodings[0]
                    sequential_index += 1
                else:
                    print(f"Предупреждение: Не найдено лиц на изображении {os.path.basename(image_path)}")
                    # Удаляем запись, если лицо не найдено
                    del self.image_encodings[sequential_index]
            except Exception as e:
                print(f"Ошибка при обработке {os.path.basename(image_path)}: {e}")
                # Удаляем запись в случае ошибки
                if sequential_index in self.image_encodings:
                    del self.image_encodings[sequential_index]

    def compare_two_faces(self, index1, index2):
        data1 = self.image_encodings.get(index1)
        data2 = self.image_encodings.get(index2)
        if data1 is None or data2 is None or data1['encoding'] is None or data2['encoding'] is None:
            # Если одно из изображений не имеет кодировки, считаем их несовпадающими
            return [False, 1.0]  # Максимальное расстояние
        encoding1 = data1['encoding']
        encoding2 = data2['encoding']
        results = face_recognition.compare_faces([encoding1], encoding2)
        face_distance = face_recognition.face_distance([encoding1], encoding2)
        return [results[0], face_distance[0]]


class SimilarityMatrix:
    def __init__(self, directory_path):
        self.face_comparator = FaceComparator()
        self.face_comparator.load(directory_path)


class MatrixPrinter:
    def __init__(self, similarity_matrix_instance):
        # Получаем количество изображений с кодировкой
        self.num_images = len(similarity_matrix_instance.face_comparator.image_encodings)
        self.similarity_matrix = [[None] * self.num_images for _ in range(self.num_images)]
        self.face_comparator = similarity_matrix_instance.face_comparator
        # Создаем список путей для удобства доступа по индексу
        self.image_paths = [self.face_comparator.image_encodings[i]['path'] for i in range(self.num_images)]

    def fill(self, start_row=0, end_row=None):
        if end_row is None:
            end_row = self.num_images
        start_row = max(0, start_row)
        end_row = min(self.num_images, end_row)
        for i in range(start_row, end_row):
            for j in range(i, self.num_images):  # Заполняем верхний треугольник и диагональ
                if i != j:
                    similarity = self.face_comparator.compare_two_faces(i, j)
                    # Заполняем симметрично
                    self.similarity_matrix[i][j] = similarity
                    self.similarity_matrix[j][i] = similarity
                else:
                    # Диагональ: изображение совпадает с самим собой
                    self.similarity_matrix[i][j] = [True, 0.0]

    def get_matrix(self):
        return self.similarity_matrix

    def print_matrix(self):
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
            group_data = {
                "id": i + 1,  # ID теперь соответствует новому порядку
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
        return final_groups_data  # Возвращаем подготовленные и отсортированные данные

    def print_groups(self):
        start_time = time.time()
        # groups_data теперь содержит подготовленные данные для JSON
        groups_data = self.group_images()
        end_time = time.time()
        grouping_time = end_time - start_time
        for group_data in groups_data:
            print(f"Группа {group_data['id']} (представлена {group_data['representative']}):")
            for path in group_data['images']:
                print(f"  {path}")
            print()
        print(f"Общее время группировки: {grouping_time:.2f} секунд")
        print(f"Найдено групп: {len(groups_data)}")
        return groups_data  # Возвращаем данные


# --- Новый класс GroupOrganizer ---
class GroupOrganizer:
    """Класс для организации файлов по группам в отдельные каталоги."""

    def __init__(self, groups_data, destination_directory):
        """
        Инициализирует GroupOrganizer.

        Args:
            groups_data (list): Список словарей с данными о группах, полученный от ImageGrouper.
            destination_directory (str): Путь к каталогу, где будут созданы подкаталоги для групп.
        """
        self.groups_data = groups_data
        self.destination_directory = destination_directory
        # Создаем основную директорию, если она не существует
        os.makedirs(self.destination_directory, exist_ok=True)

    def organize(self):
        """
        Создает каталоги для каждой группы и копирует в них файлы.
        Каталог называется по имени представителя группы.
        """
        print("=== Начало организации файлов по группам ===")
        start_time = time.time()
        total_copied = 0

        for group_data in self.groups_data:  # <-- Исправлено: было self.groups_
            group_id = group_data['id']
            # Используем имя файла представителя (без расширения) как имя каталога
            representative_name = os.path.splitext(group_data['representative'])[0]
            # Очищаем имя от недопустимых символов для имен файлов/каталогов
            safe_group_name = "".join(c for c in representative_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            # Если имя оказалось пустым, используем ID группы
            if not safe_group_name:
                safe_group_name = f"Group_{group_id}"

            group_directory_path = os.path.join(self.destination_directory, safe_group_name)
            print(f"Создаю каталог для группы {group_id}: {group_directory_path}")

            # Создаем каталог для группы
            try:
                os.makedirs(group_directory_path, exist_ok=True)
            except OSError as e:
                print(f"Ошибка создания каталога {group_directory_path}: {e}")
                continue  # Пропускаем эту группу, если не удалось создать каталог

            # Копируем файлы группы в созданный каталог
            copied_count = 0
            for full_path in group_data['images_full_paths']:
                try:
                    filename = os.path.basename(full_path)
                    destination_file_path = os.path.join(group_directory_path, filename)
                    # Копируем файл
                    shutil.copy2(full_path, destination_file_path)
                    # print(f"  Скопирован файл: {filename}") # Опционально: логировать каждый скопированный файл
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
    parser = argparse.ArgumentParser(description='Анализ и группировка изображений по схожести лиц')
    parser.add_argument('-s', '--src', required=True, help='Путь к директории с изображениями для анализа')
    # Добавляем аргумент для указания выходного файла
    parser.add_argument('-o', '--output', default='groups.json',
                        help='Путь к выходному JSON файлу (по умолчанию groups.json)')
    # Добавляем аргумент для указания директории для организованных файлов
    parser.add_argument('-d', '--dest',
                        help='Путь к директории для создания подкаталогов с группами (если не указан, организация файлов выполняться не будет)')
    args = parser.parse_args()

    if not os.path.exists(args.src):
        print(f"Ошибка: Директория '{args.src}' не существует")
        return
    if not os.path.isdir(args.src):
        print(f"Ошибка: '{args.src}' не является директорией")
        return

    # Если указана директория назначения, проверяем её или создаём
    if args.dest:
        try:
            os.makedirs(args.dest, exist_ok=True)
        except OSError as e:
            print(f"Ошибка при работе с директорией назначения '{args.dest}': {e}")
            return

    print("=== Начало анализа изображений ===")
    total_start_time = time.time()
    similarity_matrix_instance = SimilarityMatrix(args.src)
    # Проверяем, были ли загружены изображения с кодировкой
    num_loaded_images = len(similarity_matrix_instance.face_comparator.image_encodings)
    if num_loaded_images == 0:
        print("Нет изображений с лицами для анализа.")
        # Создаем пустой JSON файл
        empty_result = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_groups": 0,
            "unrecognized_count": 0,
            "groups": []
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(empty_result, f, ensure_ascii=False, indent=4)
        print(f"Создан пустой файл результатов: {args.output}")
        sys.exit(0)
    else:
        print(f"Успешно загружено лиц: {num_loaded_images}")
    matrix_printer = MatrixPrinter(similarity_matrix_instance)
    # Разделение на 4 части и вызов fill() 4 раза
    # num_encodings = len(similarity_matrix_instance.face_comparator.image_encodings)
    num_encodings = matrix_printer.num_images  # Используем правильное количество
    if num_encodings == 0:
        print("Нет изображений для сравнения после загрузки.")
        return
    chunk_size = max(1, num_encodings // 4)  # Обеспечиваем минимальный размер порции
    print(f"Размер порции для обработки: {chunk_size}")
    for i in range(4):
        start_row = i * chunk_size
        if i == 3:  # Последняя итерация обрабатывает остаток
            end_row = num_encodings
        else:
            end_row = min((i + 1) * chunk_size, num_encodings)  # Убедиться, что не выходим за границы
        if start_row >= num_encodings:
            break  # Избегаем лишних итераций, если изображений меньше 4
        print(f"Обрабатываю порцию {i + 1}/4: строки {start_row} до {end_row}")
        matrix_printer.fill(start_row, end_row)
    # print("Матрица схожести:")
    # matrix_printer.print_matrix()
    # Передаем правильные пути из matrix_printer
    grouper = ImageGrouper(matrix_printer.get_matrix(), matrix_printer.image_paths)
    # Получаем данные групп
    groups_data = grouper.print_groups()

    # --- Новый блок: Организация файлов ---
    if args.dest and groups_data:
        organizer = GroupOrganizer(groups_data, args.dest)
        organizer.organize()
    elif args.dest:
        print("Организация файлов не будет выполнена, так как не найдено групп.")
    # --- Конец нового блока ---

    # Подготавливаем данные для JSON, включая нераспознанные изображения
    total_elapsed_time = time.time() - total_start_time
    # Определяем нераспознанные изображения (те, которые не вошли ни в одну группу)
    all_indices = set(range(num_loaded_images))
    used_indices_in_groups = set()
    for group_data in groups_data:
        # Извлекаем индексы из полных путей
        for path in group_data['images_full_paths']:
            # Находим индекс по пути
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
    # Формируем итоговый JSON
    result_json = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_groups": len(groups_data),
        "unrecognized_count": len(unrecognized_data),
        "groups": groups_data,
        "unrecognized_images": unrecognized_data  # Добавлено
    }
    # Сохраняем в файл
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
