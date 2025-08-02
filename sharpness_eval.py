import json
import os
from typing import List, Tuple, Dict, Any
import cv2


class ImageSharpnessAnalyzer:
    """Класс для анализа резкости лица на изображениях"""
    # Загружаем каскад для обнаружения лиц один раз при инициализации класса
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    @staticmethod
    def calculate_face_sharpness(image_path: str) -> float:
        """
        Вычисляет резкость лица на изображении методом Лапласиана
        Args:
            image_path (str): Путь к изображению
        Returns:
            float: Значение резкости лица (дисперсия Лапласиана)
        """
        image = cv2.imread(image_path)
        if image is None:
            return 0.0

        # Проверка, загружен ли каскад
        if ImageSharpnessAnalyzer.face_cascade.empty():
            print("Ошибка: Не удалось загрузить каскад для обнаружения лиц")
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Обнаружение лиц на изображении
        faces = ImageSharpnessAnalyzer.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Если лица не найдены, возвращаем 0.0
        if len(faces) == 0:
            return 0.0

        # Выбираем наибольшее лицо (по площади)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        # Вырезаем область лица
        face_roi = gray[y:y + h, x:x + w]

        # Вычисляем резкость только для области лица
        laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
        sharpness = laplacian.var()
        return sharpness

    @staticmethod
    def get_image_info(image_path: str) -> Tuple[int, int, int, bool]:
        """
        Получает информацию об изображении и наличии лица
        Args:
            image_path (str): Путь к изображению
        Returns:
            Tuple[int, int, int, bool]: Ширина, высота, площадь, наличие лица
        """
        image = cv2.imread(image_path)
        if image is None:
            return 0, 0, 0, False

        height, width = image.shape[:2]
        area = width * height

        # Проверяем наличие лица
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = ImageSharpnessAnalyzer.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        has_face = len(faces) > 0
        return width, height, area, has_face


class ImageData:
    """Класс для хранения данных об изображении"""

    def __init__(self, filename: str, full_path: str, sharpness: float, area: int,
                 width: int, height: int, has_face: bool):
        self.filename = filename
        self.full_path = full_path
        self.sharpness = sharpness
        self.area = area
        self.width = width
        self.height = height
        self.has_face = has_face  # Новое поле для отметки наличия лица

    def __repr__(self) -> str:
        return f"ImageData(filename='{self.filename}', sharpness={self.sharpness:.2f}, " \
               f"area={self.area}, width={self.width}, height={self.height}, has_face={self.has_face})"

    def to_dict(self) -> Dict[str, Any]:
        """Возвращает данные в виде словаря для JSON"""
        return {
            "filename": self.filename,
            "full_path": self.full_path,
            "sharpness": round(self.sharpness, 2),
            "area": self.area,
            "width": self.width,
            "height": self.height,
            "has_face": self.has_face
        }


class SharpnessGrouper:
    """Класс для группировки изображений по резкости"""

    def __init__(self, images_data: List[ImageData]):
        self.images_data = images_data
        # Определяем пороги для групп
        self.thresholds = [80, 50, 25]
        self.group_names = [
            f"Группа 1: Резкость >= {self.thresholds[0]}",
            f"Группа 2: {self.thresholds[1]} <= Резкость < {self.thresholds[0]}",
            f"Группа 3: {self.thresholds[2]} <= Резкость < {self.thresholds[1]}",
            f"Группа 4: Резкость < {self.thresholds[2]}"
        ]

    def group_images(self) -> Dict[str, List[ImageData]]:
        """
        Группирует изображения по резкости.
        Returns:
            Dict[str, List[ImageData]]: Словарь с группами изображений.
        """
        groups: Dict[str, List[ImageData]] = {name: [] for name in self.group_names}
        for image_data in self.images_data:
            sharpness = image_data.sharpness
            if sharpness >= self.thresholds[0]:
                groups[self.group_names[0]].append(image_data)
            elif sharpness >= self.thresholds[1]:
                groups[self.group_names[1]].append(image_data)
            elif sharpness >= self.thresholds[2]:
                groups[self.group_names[2]].append(image_data)
            else:
                groups[self.group_names[3]].append(image_data)
        return groups

    def print_groups(self, groups: Dict[str, List[ImageData]]) -> None:
        """Выводит группы изображений в консоль."""
        print("\n--- Группировка по резкости лица ---")
        for group_name, images in groups.items():
            print(f"\n{group_name} (Количество: {len(images)}):")
            if images:
                # Печатаем только первые 5 изображений из группы для краткости
                for img in images[:5]:
                    status = "с лицом" if img.has_face else "без лица"
                    print(f"  - {img.filename} (Резкость: {img.sharpness:.2f}, {status})")
                if len(images) > 5:
                    print(f"  ... и ещё {len(images) - 5} изображений.")
            else:
                print("  Нет изображений в этой группе.")

    def get_groups_for_json(self, groups: Dict[str, List[ImageData]]) -> List[Dict[str, Any]]:
        """Подготавливает данные групп для сохранения в JSON."""
        json_groups = []
        for i, (group_name, images) in enumerate(groups.items(), start=1):
            json_group = {
                "group_id": i,
                "name": group_name,
                "size": len(images),
                "images": [img.to_dict() for img in images]
            }
            json_groups.append(json_group)
        return json_groups


class ImageProcessor:
    """Класс для обработки изображений в директории"""

    def __init__(self, directory: str):
        self.directory = directory
        self.images_data: List[ImageData] = []

    def process_images(self) -> None:
        """Обрабатывает все изображения в директории"""
        self.images_data.clear()
        for filename in os.listdir(self.directory):
            filepath = os.path.join(self.directory, filename)
            if (os.path.isfile(filepath) and
                    filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))):
                sharpness = ImageSharpnessAnalyzer.calculate_face_sharpness(filepath)
                width, height, area, has_face = ImageSharpnessAnalyzer.get_image_info(filepath)
                image_data = ImageData(filename, filepath, sharpness, area, width, height, has_face)
                self.images_data.append(image_data)

    def sort_images(self) -> None:
        """Сортирует изображения по резкости лица (убывание), затем по площади (убывание)"""
        # Сначала сортируем по наличию лица (сначала изображения с лицом), затем по резкости и площади
        self.images_data.sort(key=lambda x: (-1 if x.has_face else 1, -x.sharpness, -x.area))

    def get_sorted_data(self) -> List[ImageData]:
        """Возвращает отсортированные данные в виде списка ImageData"""
        return self.images_data


class ConsoleOutput:
    """Класс для вывода результатов в консоль"""

    @staticmethod
    def print_header() -> None:
        """Выводит заголовок таблицы"""
        print(f"{'Имя файла':<30} {'Резкость':<15} {'Лицо':<10} {'Площадь':<10} {'Ширина':<8} {'Высота':<8}")
        print("-" * 95)

    @staticmethod
    def print_image_data(image_data: ImageData) -> None:
        """Выводит данные об одном изображении"""
        # Ограничиваем длину имени файла для лучшего форматирования
        short_name = (image_data.filename[:27] + '...') if len(image_data.filename) > 30 else image_data.filename
        has_face_str = "Да" if image_data.has_face else "Нет"
        print(
            f"{short_name:<30} {image_data.sharpness:<15.2f} {has_face_str:<10} {image_data.area:<10} {image_data.width:<8} {image_data.height:<8}")

    @staticmethod
    def print_results(images_data: List[ImageData]) -> None:
        """Выводит все результаты"""
        ConsoleOutput.print_header()
        for image_data in images_data:
            ConsoleOutput.print_image_data(image_data)


class SharpnessEvaluator:
    """Основной класс для оценки резкости изображений"""

    def __init__(self, directory: str):
        self.directory = directory
        self.processor = ImageProcessor(directory)
        self.output = ConsoleOutput()

    def run(self, input_json_path: str = None) -> None:
        """Запускает полный процесс оценки резкости"""
        if not os.path.isdir(self.directory):
            print(f"Ошибка: Директория '{self.directory}' не существует.")
            return

        print(f"Обработка изображений в директории: {self.directory}")

        # Обрабатываем изображения
        self.processor.process_images()

        # Сортируем результаты
        self.processor.sort_images()

        # Получаем отсортированные данные
        sorted_data = self.processor.get_sorted_data()

        # Выводим результаты
        if sorted_data:
            print("\n--- Все изображения, отсортированные по четкости лица ---")
            self.output.print_results(sorted_data)
        else:
            print("В указанной директории не найдено изображений.")
            return  # Если нет изображений, выходим

        # --- Новая логика группировки ---
        grouper = SharpnessGrouper(sorted_data)
        sharpness_groups = grouper.group_images()
        grouper.print_groups(sharpness_groups)

        # --- Подготовка и сохранение JSON ---
        json_output = {}

        # Если предоставлен входной JSON, загружаем его
        if input_json_path and os.path.isfile(input_json_path):
            try:
                with open(input_json_path, 'r', encoding='utf-8') as f:
                    json_output = json.load(f)
                print(f"\nЗагружен входной JSON из файла: {input_json_path}")
            except Exception as e:
                print(f"Ошибка при загрузке входного JSON: {e}. Создаётся новый файл.")

        # Добавляем данные о группировке по резкости в JSON
        json_output["sharpness_groups"] = grouper.get_groups_for_json(sharpness_groups)

        # --- Сохраняем JSON в ту же директорию, что и изображения ---
        output_json_path = os.path.join(self.directory, "output_with_face_sharpness_groups.json")
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, ensure_ascii=False, indent=4)
            print(f"\nРезультаты, включая группы по четкости лица, сохранены в файл: {output_json_path}")
        except Exception as e:
            print(f"Ошибка при сохранении JSON: {e}")


def main():
    """Основная функция"""
    import sys
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Использование: python script.py <директория_с_изображениями> [путь_к_input.json]")
        return

    directory = sys.argv[1]
    input_json_path = sys.argv[2] if len(sys.argv) == 3 else None

    evaluator = SharpnessEvaluator(directory)
    # Передаём пути к JSON файлам
    evaluator.run(input_json_path=input_json_path)


if __name__ == "__main__":
    main()