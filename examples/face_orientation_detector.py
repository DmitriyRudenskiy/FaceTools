import face_recognition
import cv2
import numpy as np
import os
from pathlib import Path
import sys
import argparse


class FaceOrientationDetector:
    def __init__(self):
        # Английские псевдонимы для ориентаций
        self.orientation_aliases = {
            "front": "front",  # Анфас
            "profile_left": "profile_left",  # Профиль влево
            "profile_right": "profile_right",  # Профиль вправо
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

            # Проверяем наличие ключевых точек глаз
            left_eye = landmarks.get('left_eye')
            right_eye = landmarks.get('right_eye')

            # Если оба глаза видны
            if left_eye and right_eye:
                left_eye_mean = np.mean(left_eye, axis=0)
                right_eye_mean = np.mean(right_eye, axis=0)

                # Вычисляем угол между глазами
                delta_x = right_eye_mean[0] - left_eye_mean[0]
                delta_y = right_eye_mean[1] - left_eye_mean[1]
                angle = np.degrees(np.arctan2(delta_y, delta_x))

                # Определяем ориентацию
                if abs(angle) < 15:
                    return "front"
                elif angle > 15:
                    return "profile_right"
                elif angle < -15:
                    return "profile_left"
                else:
                    return "semi_front"

            # Если только один глаз виден, используем его положение
            elif left_eye:
                eye = np.mean(left_eye, axis=0)
                nose = np.mean(landmarks['nose_bridge'], axis=0)
                delta_x = nose[0] - eye[0]
                delta_y = nose[1] - eye[1]
                angle = np.degrees(np.arctan2(delta_y, delta_x))

                # Определяем ориентацию
                if angle > 30:
                    return "profile_right"
                elif angle < -30:
                    return "profile_left"
                else:
                    return "semi_front"

            elif right_eye:
                eye = np.mean(right_eye, axis=0)
                nose = np.mean(landmarks['nose_bridge'], axis=0)
                delta_x = nose[0] - eye[0]
                delta_y = nose[1] - eye[1]
                angle = np.degrees(np.arctan2(delta_y, delta_x))

                # Определяем ориентацию
                if angle > 30:
                    return "profile_right"
                elif angle < -30:
                    return "profile_left"
                else:
                    return "semi_front"

            # Если глаза не видны, используем положение носа и уха
            else:
                nose = np.mean(landmarks['nose_bridge'], axis=0)
                ear = np.mean(landmarks['left_ear'], axis=0) if 'left_ear' in landmarks else np.mean(
                    landmarks['right_ear'], axis=0)

                delta_x = ear[0] - nose[0]
                delta_y = ear[1] - nose[1]
                angle = np.degrees(np.arctan2(delta_y, delta_x))

                # Определяем ориентацию
                if angle > 30:
                    return "profile_right"
                elif angle < -30:
                    return "profile_left"
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
                    print(f"⚠️  Не удалось обнаружить лицо или определить его ориентацию: {file_path.name}")
                    continue

                alias = self.get_alias(orientation)

                # Если файл уже имеет префикс, пропускаем
                if self.has_alias_prefix(file_path.name):
                    processed_count += 1
                    continue

                # Создаем новое имя файла
                new_filename = f"{alias}_{file_path.name}"
                new_file_path = file_path.parent / new_filename

                # Переименовываем файл
                try:
                    file_path.rename(new_file_path)
                    renamed_count += 1
                except Exception as e:
                    print(f"❌ Ошибка переименования {file_path.name}: {e}")

                processed_count += 1

        print(f"\n📊 Результаты обработки:")
        print(f"   Обработано файлов: {processed_count}")
        print(f"   Переименовано файлов: {renamed_count}")


def main():
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(
        description='Face Orientation Detector - определяет ориентацию лица на фотографиях и переименовывает файлы')
    parser.add_argument('directory', help='Путь к директории с изображениями')

    # Парсим аргументы
    args = parser.parse_args()

    # Создаем экземпляр детектора
    detector = FaceOrientationDetector()

    # Обрабатываем директорию, переданную как аргумент
    detector.process_directory(args.directory)


# Запуск скрипта
if __name__ == "__main__":
    main()