import os
import cv2
import numpy as np
from typing import List, Tuple

class ImageSharpnessAnalyzer:
    """Класс для анализа резкости изображений"""
    
    @staticmethod
    def calculate_sharpness(image_path: str) -> float:
        """
        Вычисляет резкость изображения методом Лапласиана
        
        Args:
            image_path (str): Путь к изображению
            
        Returns:
            float: Значение резкости (дисперсия Лапласиана)
        """
        image = cv2.imread(image_path)
        if image is None:
            return 0.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        return sharpness
    
    @staticmethod
    def get_image_info(image_path: str) -> Tuple[int, int, int]:
        """
        Получает информацию об изображении
        
        Args:
            image_path (str): Путь к изображению
            
        Returns:
            Tuple[int, int, int]: Ширина, высота, площадь
        """
        image = cv2.imread(image_path)
        if image is None:
            return 0, 0, 0
        
        height, width = image.shape[:2]
        area = width * height
        return width, height, area

class ImageData:
    """Класс для хранения данных об изображении"""
    
    def __init__(self, filename: str, sharpness: float, area: int, width: int, height: int):
        self.filename = filename
        self.sharpness = sharpness
        self.area = area
        self.width = width
        self.height = height
    
    def __repr__(self) -> str:
        return f"ImageData(filename='{self.filename}', sharpness={self.sharpness:.2f}, area={self.area}, width={self.width}, height={self.height})"
    
    def to_tuple(self) -> Tuple[str, float, int, int, int]:
        """Возвращает данные в виде кортежа"""
        return (self.filename, self.sharpness, self.area, self.width, self.height)

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
                
                sharpness = ImageSharpnessAnalyzer.calculate_sharpness(filepath)
                width, height, area = ImageSharpnessAnalyzer.get_image_info(filepath)
                
                image_data = ImageData(filename, sharpness, area, width, height)
                self.images_data.append(image_data)
    
    def sort_images(self) -> None:
        """Сортирует изображения по резкости (убывание), затем по площади (убывание)"""
        self.images_data.sort(key=lambda x: (-x.sharpness, -x.area))
    
    def get_sorted_data(self) -> List[Tuple[str, float, int, int, int]]:
        """Возвращает отсортированные данные в виде списка кортежей"""
        return [image_data.to_tuple() for image_data in self.images_data]

class ConsoleOutput:
    """Класс для вывода результатов в консоль"""
    
    @staticmethod
    def print_header() -> None:
        """Выводит заголовок таблицы"""
        print(f"{'Имя файла':<30} {'Резкость':<15} {'Площадь':<10} {'Ширина':<8} {'Высота':<8}")
        print("-" * 75)
    
    @staticmethod
    def print_image_data(image_data: Tuple[str, float, int, int, int]) -> None:
        """Выводит данные об одном изображении"""
        filename, sharpness, area, width, height = image_data
        print(f"{filename:<30} {sharpness:<15.2f} {area:<10} {width:<8} {height:<8}")
    
    @staticmethod
    def print_results(images_data: List[Tuple[str, float, int, int, int]]) -> None:
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
    
    def run(self) -> None:
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
            self.output.print_results(sorted_data)
        else:
            print("В указанной директории не найдено изображений.")

def main():
    """Основная функция"""
    import sys
    
    if len(sys.argv) != 2:
        print("Использование: python script.py <директория_с_изображениями>")
        return
    
    directory = sys.argv[1]
    evaluator = SharpnessEvaluator(directory)
    evaluator.run()

if __name__ == "__main__":
    main()