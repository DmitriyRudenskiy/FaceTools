import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceComparator:
    # Переменная класса для хранения изображений и их эмбеддингов (массив)
    storage = []

    def __init__(self):
        self.device = torch.device('cpu')

        # Инициализация MTCNN для обнаружения и выравнивания лиц
        self.mtcnn = MTCNN(
            image_size=512,
            margin=0,
            min_face_size=384,
            thresholds=[0.5, 0.5, 0.5],
            factor=0.5,
            post_process=True,
            device=self.device
        )

        # Инициализация предобученной модели FaceNet
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def init(self, image_paths):
        """
        Инициализировать хранилище массивом путей к изображениям

        Args:
            image_paths: список путей к изображениям

        Returns:
            list: список индексов добавленных изображений
        """
        # Очистка существующего хранилища
        self.clear_storage()

        indices = []
        for image_path in image_paths:
            try:
                index = self._add_single_image(image_path)
                indices.append(index)
                print(f"✅ Изображение {image_path} добавлено под индексом {index}")
            except Exception as e:
                print(f"❌ Ошибка при добавлении изображения {image_path}: {str(e)}")

        return indices

    def _add_single_image(self, image_path):
        """
        Добавить одно изображение в хранилище (внутренний метод)

        Args:
            image_path: путь к изображению

        Returns:
            int: индекс добавленного изображения
        """
        # Загрузка и обработка изображения
        img = Image.open(image_path).convert('RGB')
        face = self.mtcnn(img)

        if face is None:
            raise ValueError(f"Лицо не обнаружено на изображении {image_path}")

        # Добавление размерности батча и перемещение на устройство
        face = face.unsqueeze(0).to(self.device)

        # Генерация эмбеддинга
        with torch.no_grad():
            embedding = self.resnet(face)

        # Создание записи для хранения
        image_record = {
            'image_path': image_path,
            'embedding': embedding,
            'face_tensor': face
        }

        # Добавление в массив и получение индекса
        self.storage.append(image_record)
        index = len(self.storage) - 1

        return index

    def add_image(self, image_path):
        """
        Добавить отдельное изображение в хранилище

        Args:
            image_path: путь к изображению

        Returns:
            int: индекс добавленного изображения
        """
        try:
            index = self._add_single_image(image_path)
            print(f"✅ Изображение добавлено в хранилище под индексом {index}")
            return index
        except Exception as e:
            raise ValueError(f"Ошибка при добавлении изображения: {str(e)}")

    def compare_by_index(self, index1, index2, threshold=0.7):
        """
        Сравнить два изображения по их индексам в хранилище

        Args:
            index1: индекс первого изображения
            index2: индекс второго изображения
            threshold: порог для определения совпадения (по умолчанию 0.7)

        Returns:
            dict: результат сравнения с расстоянием и выводом
        """
        # Проверка корректности индексов
        if index1 < 0 or index1 >= len(self.storage):
            raise ValueError(f"Индекс {index1} выходит за границы хранилища")

        if index2 < 0 or index2 >= len(self.storage):
            raise ValueError(f"Индекс {index2} выходит за границы хранилища")

        # Получение эмбеддингов
        embedding1 = self.storage[index1]['embedding']
        embedding2 = self.storage[index2]['embedding']

        # Вычисление евклидова расстояния между эмбеддингами
        distance = (embedding1 - embedding2).norm().item()

        # Определение результата
        is_same_person = distance < threshold

        result = {
            'distance': distance,
            'is_same_person': is_same_person,
            'threshold': threshold,
            'indices': (index1, index2),
            'message': "✅ Лица принадлежат одному человеку" if is_same_person else "❌ Лица принадлежат разным людям"
        }

        return result

    def compare_image_with_index(self, image_path, index, threshold=0.7):
        """
        Сравнить новое изображение с изображением из хранилища по индексу

        Args:
            image_path: путь к новому изображению
            index: индекс изображения в хранилище
            threshold: порог для определения совпадения

        Returns:
            dict: результат сравнения
        """
        if index < 0 or index >= len(self.storage):
            raise ValueError(f"Индекс {index} выходит за границы хранилища")

        try:
            # Загрузка и обработка нового изображения
            img = Image.open(image_path).convert('RGB')
            face = self.mtcnn(img)

            if face is None:
                raise ValueError(f"Лицо не обнаружено на изображении {image_path}")

            # Добавление размерности батча и перемещение на устройство
            face = face.unsqueeze(0).to(self.device)

            # Генерация эмбеддинга для нового изображения
            with torch.no_grad():
                embedding_new = self.resnet(face)

            # Получение эмбеддинга из хранилища
            embedding_stored = self.storage[index]['embedding']

            # Вычисление евклидова расстояния
            distance = (embedding_new - embedding_stored).norm().item()

            # Определение результата
            is_same_person = distance < threshold

            result = {
                'distance': distance,
                'is_same_person': is_same_person,
                'threshold': threshold,
                'storage_index': index,
                'message': "✅ Лица принадлежат одному человеку" if is_same_person else "❌ Лица принадлежат разным людям"
            }

            return result

        except Exception as e:
            raise ValueError(f"Ошибка при сравнении изображений: {str(e)}")

    @classmethod
    def get_storage_info(cls):
        """
        Получить информацию о содержимом хранилища

        Returns:
            dict: информация о хранилище
        """
        return {
            'total_images': len(cls.storage),
            'indices': list(range(len(cls.storage)))
        }

    @classmethod
    def clear_storage(cls):
        """
        Очистить хранилище
        """
        cls.storage.clear()
        print("✅ Хранилище очищено")

    def get_image_info(self, index):
        """
        Получить информацию об изображении по индексу

        Args:
            index: индекс изображения

        Returns:
            dict: информация об изображении
        """
        if index < 0 or index >= len(self.storage):
            raise ValueError(f"Индекс {index} выходит за границы хранилища")

        return {
            'index': index,
            'image_path': self.storage[index]['image_path']
        }


# Пример использования
if __name__ == "__main__":
    # Создание экземпляра класса
    comparator = FaceComparator()

    try:
        # Инициализация хранилища массивом путей к изображениям
        image_paths = [
            "/Users/user/__!make_face/refer_bibi_1.png",
            "/Users/user/__!make_face/refer_bibi_2.png"
        ]

        indices = comparator.init(image_paths)
        print(f"Изображения добавлены под индексами: {indices}")

        # Сравнение изображений по индексам
        if len(indices) >= 2:
            result = comparator.compare_by_index(indices[0], indices[1])

            print(f"\nЕвклидово расстояние между эмбеддингами: {result['distance']:.4f}")
            print(result['message'])
            print(f"Порог: {result['threshold']}")
            print(f"Сравнивались индексы: {result['indices']}")

        # Получение информации о хранилище
        storage_info = comparator.get_storage_info()
        print(f"\nХранилище содержит {storage_info['total_images']} изображений")
        print(f"Доступные индексы: {storage_info['indices']}")

    except Exception as e:
        print(f"Ошибка: {str(e)}")