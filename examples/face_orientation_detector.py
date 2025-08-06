import face_recognition
import cv2
import numpy as np
import os
from pathlib import Path


class FaceOrientationDetector:
    def __init__(self):
        # Английские псевдонимы для ориентаций
        self.orientation_aliases = {
            "front": "front",  # Анфас
            "left_profile": "l_prof",  # Профиль влево
            "right_profile": "r_prof",  # Профиль вправо
            "semi_front": "semi"  # Полупрофиль
        }

    def detect_orientation(self, image_path):
        """
        Определяет ориентацию лица на изображении
        """
        try:
            # Загружаем изображение
            image = face_recognition.load_image_file(image_path)
            face_landmarks_list = face_recognition.face_landmarks(image)

            if not face_landmarks_list:
                return None

            # Берем первое найденное лицо
            landmarks = face_landmarks_list[0]

            # Получаем ключевые точки глаз
            left_eye = np.mean(landmarks['left_eye'], axis=0)
            right_eye = np.mean(landmarks['right_eye'], axis=0)

            # Вычисляем угол между глазами
            delta_x = right_eye[0] - left_eye[0]
            delta_y = right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(delta_y, delta_x))

            # Определяем ориентацию
            if abs(angle) < 15:
                return "front"
            elif angle > 15:
                return "right_profile"
            elif angle < -15:
                return "left_profile"
            else:
                return "semi_front"

        except Exception as e:
            print(f"Ошибка при обработке файла {image_path}: {e}")
            return None

    def get_alias(self, orientation):
        """
        Возвращает английский псевдоним для ориентации
        """
        return self.orientation_aliases.get(orientation, orientation)

    def has_alias_prefix(self, filename):
        """
        Проверяет, есть ли уже псевдоним в начале имени файла
        """
        for alias in self.orientation_aliases.values():
            if filename.startswith(f"{alias}_"):
                return True
        return False

    def process_directory(self, directory_path):
        """
        Обрабатывает все изображения в директории
        """
        directory = Path(directory_path)

        if not directory.exists():
            print(f"❌ Директория {directory_path} не существует")
            return

        # Поддерживаемые форматы изображений
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        processed_count = 0
        renamed_count = 0

        print(f"🔍 Начинаем обработку директории: {directory_path}")

        # Проходим по всем файлам в директории
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                print(f"🔄 Обрабатываем: {file_path.name}")

                # Определяем ориентацию
                orientation = self.detect_orientation(str(file_path))

                if orientation is None:
                    print(f"⚠️  Не удалось определить ориентацию: {file_path.name}")
                    continue

                alias = self.get_alias(orientation)

                # Если файл уже имеет префикс, пропускаем
                if self.has_alias_prefix(file_path.name):
                    print(f"✅ Файл уже имеет префикс: {file_path.name}")
                    processed_count += 1
                    continue

                # Создаем новое имя файла
                new_filename = f"{alias}_{file_path.name}"
                new_file_path = file_path.parent / new_filename

                # Переименовываем файл
                try:
                    file_path.rename(new_file_path)
                    print(f"✅ Переименован: {file_path.name} → {new_filename}")
                    renamed_count += 1
                except Exception as e:
                    print(f"❌ Ошибка переименования {file_path.name}: {e}")

                processed_count += 1

        print(f"\n📊 Результаты обработки:")
        print(f"   Обработано файлов: {processed_count}")
        print(f"   Переименовано файлов: {renamed_count}")


# Пример использования
if __name__ == "__main__":
    # Создаем экземпляр детектора
    detector = FaceOrientationDetector()

    # Укажите путь к директории с изображениями
    directory_path = input("📁 Введите путь к директории с изображениями: ").strip()

    # Обрабатываем директорию
    detector.process_directory(directory_path)