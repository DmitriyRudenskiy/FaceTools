import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FaceComparator:
    def __init__(self, device=None, threshold=0.7):
        """
        Инициализация компаратора лиц

        Изменения:
        1. Автоматическое определение устройства (GPU/CPU)
        2. Параметр порога по умолчанию
        3. Инициализация хранилища в экземпляре
        4. Настройка логирования
        """
        # Автоматическое определение устройства
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device
        logger.info(f"Используется устройство: {self.device}")

        # Настройка порога по умолчанию
        self.default_threshold = threshold
        logger.info(f"Порог по умолчанию: {threshold}")

        # Инициализация MTCNN с оптимизированными параметрами
        self.mtcnn = MTCNN(
            image_size=160,  # Стандартный размер для FaceNet
            margin=40,
            min_face_size=60,
            thresholds=[0.6, 0.7, 0.7],  # Более строгие пороги
            factor=0.709,
            post_process=True,
            device=self.device
        )

        # Инициализация модели ResNet
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Хранилище в экземпляре класса (вместо классовой переменной)
        self.storage = []

    def init(self, image_paths):
        """
        Пакетное добавление изображений в хранилище

        Изменения:
        1. Возвращает словарь с результатами вместо списка индексов
        2. Подробное логирование
        3. Обработка исключений для каждого изображения
        """
        results = {'success': [], 'errors': []}

        for image_path in image_paths:
            try:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Файл не найден: {image_path}")

                index = self._add_single_image(image_path)
                results['success'].append({
                    'index': index,
                    'path': image_path
                })
            except Exception as e:
                error_msg = f"❌ Ошибка при добавлении {image_path}: {str(e)}"
                results['errors'].append({
                    'path': image_path,
                    'error': str(e)
                })
                logger.error(error_msg)

        return results

    def _add_single_image(self, image_path):
        """Внутренний метод для добавления одного изображения"""
        img = Image.open(image_path).convert('RGB')

        # Обнаружение лица с обработкой исключений
        face = self.mtcnn(img)
        if face is None:
            raise ValueError("Лица не обнаружены")

        # Генерация эмбеддинга
        face = face.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.resnet(face)

        # Конвертация в numpy для экономии памяти
        embedding_np = embedding.cpu().numpy().squeeze()

        self.storage.append({
            'image_path': image_path,
            'embedding': embedding_np,
            'face_tensor': face.cpu()  # Перемещаем на CPU для экономии памяти GPU
        })

        return len(self.storage) - 1

    def add_image(self, image_path):
        """Добавление одного изображения с улучшенной обработкой ошибок"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл не найден: {image_path}")

        try:
            index = self._add_single_image(image_path)
            return index
        except Exception as e:
            raise RuntimeError(f"Ошибка добавления: {str(e)}")

    def _calculate_distance(self, emb1, emb2):
        """Расчет косинусного расстояния между эмбеддингами"""
        # Конвертация в torch при необходимости
        if isinstance(emb1, np.ndarray):
            emb1 = torch.from_numpy(emb1).to(self.device)
        if isinstance(emb2, np.ndarray):
            emb2 = torch.from_numpy(emb2).to(self.device)

        # Нормализация векторов
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=0)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=0)

        # Косинусная схожесть
        similarity = torch.dot(emb1, emb2).item()
        distance = 1.0 - similarity

        return distance, similarity

    def compare_by_index(self, index1, index2, threshold=None):
        """Сравнение по индексам с улучшенной метрикой"""
        threshold = threshold or self.default_threshold

        # Валидация индексов
        for i in [index1, index2]:
            if i < 0 or i >= len(self.storage):
                raise IndexError(f"Неверный индекс: {i}")

        emb1 = self.storage[index1]['embedding']
        emb2 = self.storage[index2]['embedding']

        distance, similarity = self._calculate_distance(emb1, emb2)
        is_same = distance < threshold

        return {
            'distance': distance,
            'similarity': similarity,
            'is_same_person': is_same,
            'threshold': threshold,
            'indices': (index1, index2)
        }

    def compare_image_with_index(self, image_path, index, threshold=None):
        """Сравнение внешнего изображения с хранилищем"""
        threshold = threshold or self.default_threshold

        if index < 0 or index >= len(self.storage):
            raise IndexError(f"Неверный индекс: {index}")

        try:
            # Обработка входного изображения
            img = Image.open(image_path).convert('RGB')
            face = self.mtcnn(img)

            if face is None:
                raise ValueError("Лица не обнаружены")

            face = face.unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding_new = self.resnet(face)

            # Сравнение эмбеддингов
            stored_emb = self.storage[index]['embedding']
            embedding_new = embedding_new.cpu().numpy().squeeze()

            distance, similarity = self._calculate_distance(embedding_new, stored_emb)
            is_same = distance < threshold

            return {
                'distance': distance,
                'similarity': similarity,
                'is_same_person': is_same,
                'threshold': threshold,
                'storage_index': index
            }
        except Exception as e:
            logger.exception("Ошибка сравнения")
            raise RuntimeError(f"Ошибка сравнения: {str(e)}")

    def get_storage_info(self):
        """Информация о хранилище"""
        return {
            'total_images': len(self.storage),
            'indices': list(range(len(self.storage))),
            'paths': [item['image_path'] for item in self.storage]
        }

    def clear_storage(self):
        """Очистка хранилища с освобождением памяти"""
        self.storage.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("Хранилище очищено")

    def remove_image(self, index):
        """Удаление изображения по индексу"""
        if 0 <= index < len(self.storage):
            removed = self.storage.pop(index)
            logger.info(f"Удалено изображение {removed['image_path']} (индекс {index})")
            return True
        raise IndexError(f"Неверный индекс: {index}")

    def compare_two_images(self, path1, path2, threshold=None):
        """Новый метод: сравнение двух изображений без добавления в хранилище"""
        threshold = threshold or self.default_threshold

        def process_image(path):
            img = Image.open(path).convert('RGB')
            face = self.mtcnn(img)
            if face is None:
                raise ValueError(f"Лица не обнаружены в {path}")
            face = face.unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.resnet(face)
            return embedding.cpu().numpy().squeeze()

        emb1 = process_image(path1)
        emb2 = process_image(path2)

        distance, similarity = self._calculate_distance(emb1, emb2)
        is_same = distance < threshold

        return {
            'distance': distance,
            'similarity': similarity,
            'is_same_person': is_same,
            'threshold': threshold
        }


# Пример использования с улучшенной обработкой
if __name__ == "__main__":
    comparator = FaceComparator(threshold=0.65)

    # Тестовые изображения
    test_images = [
        "/Users/user/__!make_face/refer_bibi_1.png",
        "/Users/user/__!make_face/refer_bibi_2.png",
        "invalid_path.jpg"
    ]

    # Пакетное добавление
    init_result = comparator.init(test_images)
    print("\nРезультаты инициализации:")
    print(f"Успешно: {len(init_result['success'])}")
    print(f"Ошибки: {len(init_result['errors'])}")

    if init_result['success']:
        # Сравнение изображений
        indices = [item['index'] for item in init_result['success']]
        if len(indices) >= 2:
            result = comparator.compare_by_index(indices[0], indices[1])
            print("\nРезультат сравнения:")
            print(f"Дистанция: {result['distance']:.4f}")
            print(f"Схожесть: {result['similarity']:.4f}")
            print(f"Один человек: {'Да' if result['is_same_person'] else 'Нет'}")
            print(f"Порог: {result['threshold']}")

        # Информация о хранилище
        storage_info = comparator.get_storage_info()
        print(f"\nВ хранилище: {storage_info['total_images']} изображений")
        print(f"Пути: {storage_info['paths']}")

        # Удаление изображения
        comparator.remove_image(0)
        print(f"\nПосле удаления: {comparator.get_storage_info()['total_images']} изображений")

    # Сравнение без добавления в хранилище
    print("\nСравнение двух внешних изображений:")
    ext_comp = comparator.compare_two_images(test_images[0], test_images[1])
    print(f"Результат: {'Один человек' if ext_comp['is_same_person'] else 'Разные люди'}")
    print(f"Дистанция: {ext_comp['distance']:.4f}")