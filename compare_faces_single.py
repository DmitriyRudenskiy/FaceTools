import os
import time
import argparse
import face_recognition
import sys
import json  # Импортируем json для сохранения результата

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
             return [False, 1.0] # Максимальное расстояние
        encoding1 = data1['encoding']
        encoding2 = data2['encoding']
        results = face_recognition.compare_faces([encoding1], encoding2)
        face_distance = face_recognition.face_distance([encoding1], encoding2)
        return [results[0], face_distance[0]]

class SimilarityMatrix:
    def __init__(self, directory_path):
        self.face_comparator = FaceComparator()
        self.face_comparator.load(directory_path)
        # image_paths теперь не нужен, так как пути хранятся в face_comparator.image_encodings
        # self.image_paths = [...] 

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
            for j in range(i, self.num_images):
                if i != j:
                    similarity = self.face_comparator.compare_two_faces(i, j)
                    self.similarity_matrix[i][j] = similarity
                    self.similarity_matrix[j][i] = similarity
                else:
                    self.similarity_matrix[i][j] = [True, 0.0]

    def get_matrix(self):
        return self.similarity_matrix

    def print_matrix(self):
        if not self.similarity_matrix or self.num_images == 0:
            print("Нет изображений для сравнения.")
            return
        element_width = 7
        header_items = [f"{i+1:5}" for i in range(self.num_images)]
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
                average_distance = float('inf')
            distances.append((average_distance, i))
        return distances

    def group_images(self):
        start_time = time.time()
        print("Начинаю группировку изображений...")
        # Исправленный алгоритм группировки: связные компоненты
        for index in range(self.num_images):
            if index in self.used_indices:
                continue
            # Начинаем новую группу
            current_group = {index}
            stack = [index] # Используем стек для DFS
            self.used_indices.add(index)

            while stack:
                current_index = stack.pop()
                # Проверяем все другие изображения
                for other_index in range(self.num_images):
                     # Пропускаем, если уже обработано или это то же изображение
                    if other_index in self.used_indices or other_index == current_index:
                        continue
                    result = self.similarity_matrix[current_index][other_index]
                    # Если изображения похожи, добавляем в группу
                    if result is not None and result[0]:
                        current_group.add(other_index)
                        self.used_indices.add(other_index)
                        stack.append(other_index) # Добавляем в стек для дальнейшего поиска

            # Добавляем найденную группу в список
            if len(current_group) > 1: # Только группы с более чем одним элементом
                 self.groups.append(list(current_group))
            # else:
            #     print(f"Изображение {self.image_paths[list(current_group)[0]]} не имеет пары и будет в 'нераспознанных'.")

        final_groups_data = []
        for i, group_indices in enumerate(self.groups):
             # Рассчитываем средние расстояния внутри группы
            distances = self.calculate_average_distance(group_indices)
            # Находим изображение с минимальным средним расстоянием (представитель)
            min_avg_distance_index = min(distances, key=lambda x: x[0])[1]
            representative_image_path = self.image_paths[min_avg_distance_index]

            # Подготавливаем данные для JSON
            group_filenames = [os.path.basename(self.image_paths[idx]) for idx in group_indices]
            group_full_paths = [self.image_paths[idx] for idx in group_indices]
            representative_filename = os.path.basename(representative_image_path)

            group_data = {
                "id": i + 1,
                "size": len(group_indices),
                "representative": representative_filename,
                "representative_full_path": representative_image_path, # Добавлено
                "images": group_filenames,
                "images_full_paths": group_full_paths # Добавлено
            }
            final_groups_data.append(group_data)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Группировка завершена за {elapsed_time:.2f} секунд")
        return final_groups_data # Возвращаем подготовленные данные

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
        return groups_data # Возвращаем данные

def main():
    parser = argparse.ArgumentParser(description='Анализ и группировка изображений по схожести лиц')
    parser.add_argument('-s', '--src', required=True, help='Путь к директории с изображениями для анализа')
    # Добавляем аргумент для указания выходного файла
    parser.add_argument('-o', '--output', default='groups.json', help='Путь к выходному JSON файлу (по умолчанию groups.json)')
    args = parser.parse_args()

    if not os.path.exists(args.src):
        print(f"Ошибка: Директория '{args.src}' не существует")
        return
    if not os.path.isdir(args.src):
        print(f"Ошибка: '{args.src}' не является директорией")
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
    num_encodings = matrix_printer.num_images # Используем правильное количество
    if num_encodings == 0:
         print("Нет изображений для сравнения после загрузки.")
         return

    chunk_size = max(1, num_encodings // 4) # Обеспечиваем минимальный размер порции

    print(f"Размер порции для обработки: {chunk_size}")

    for i in range(4):
        start_row = i * chunk_size
        if i == 3:  # Последняя итерация обрабатывает остаток
            end_row = num_encodings
        else:
            end_row = min((i + 1) * chunk_size, num_encodings) # Убедиться, что не выходим за границы

        if start_row >= num_encodings:
            break # Избегаем лишних итераций, если изображений меньше 4

        print(f"Обрабатываю порцию {i+1}/4: строки {start_row} до {end_row}")
        matrix_printer.fill(start_row, end_row)

    # print("\nМатрица схожести:")
    # matrix_printer.print_matrix()

    # Передаем правильные пути из matrix_printer
    grouper = ImageGrouper(matrix_printer.get_matrix(), matrix_printer.image_paths)
    # Получаем данные групп
    groups_data = grouper.print_groups()

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
        "unrecognized_images": unrecognized_data # Добавлено
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
    except Exception as e:
        print(f"Ошибка при сохранении файла {args.output}: {e}")

if __name__ == "__main__":
    main()