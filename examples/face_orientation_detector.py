import face_recognition
import cv2
import numpy as np
import os
from pathlib import Path
import sys
import argparse

class FaceOrientationDetector:
    def __init__(self):
        self.orientation_aliases = {
            "front": "front",
            "profile_left": "profile_left",
            "profile_right": "profile_right",
            "semi_front": "semi"
        }

    def detect_orientation(self, image_path):
        try:
            image = face_recognition.load_image_file(image_path)
            face_landmarks_list = face_recognition.face_landmarks(image)

            if not face_landmarks_list:
                return None

            landmarks = face_landmarks_list[0]
            image_center = np.array([image.shape[1]/2, image.shape[0]/2])

            # Проверка наличия ключевых точек
            left_eye = landmarks.get('left_eye')
            right_eye = landmarks.get('right_eye')
            left_ear = landmarks.get('left_ear')
            right_ear = landmarks.get('right_ear')

            # Определение ориентации через глаза
            if left_eye and right_eye:
                left_eye_center = np.mean(left_eye, axis=0)
                right_eye_center = np.mean(right_eye, axis=0)

                # Расчет вектора между глазами
                eye_vector = right_eye_center - left_eye_center
                angle = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))

                # Определение основного направления
                main_angle = np.abs(angle) % 180
                if main_angle < 10:
                    return "front"
                elif main_angle > 80:
                    return "profile_right" if eye_vector[0] > 0 else "profile_left"
                else:
                    return "semi_front"

            # Определение через один глаз и ухо
            elif (left_eye and left_ear) or (right_eye and right_ear):
                eye = np.mean(left_eye if left_eye else right_eye, axis=0)
                ear = np.mean(left_ear if left_ear else right_ear, axis=0)

                # Проверка расположения относительно центра
                eye_side = 'left' if eye[0] < image_center[0] else 'right'
                ear_side = 'left' if ear[0] < image_center[0] else 'right'

                if eye_side != ear_side:
                    return "profile_right" if eye_side == 'left' else "profile_left"
                else:
                    return "semi_front"

            # Определение через уши
            elif left_ear or right_ear:
                ear = np.mean(left_ear if left_ear else right_ear, axis=0)
                return "profile_right" if ear[0] < image_center[0] else "profile_left"

            # Падежный случай
            else:
                return "semi_front"

        except Exception as e:
            print(f"Ошибка при обработке файла {image_path}: {e}")
            return None

    def get_alias(self, orientation):
        return self.orientation_aliases.get(orientation, orientation)

    def has_alias_prefix(self, filename):
        for alias in self.orientation_aliases.values():
            if filename.startswith(f"{alias}_"):
                return True
        return False

    def process_directory(self, directory_path):
        directory = Path(directory_path)

        if not directory.exists():
            print(f"❌ Директория {directory_path} не существует")
            return

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        processed_count = 0
        renamed_count = 0

        print(f"🔍 Начинаем обработку директории: {directory_path}")

        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                print(f"🔄 Обрабатываем: {file_path.name}")

                orientation = self.detect_orientation(str(file_path))

                if orientation is None:
                    print(f"⚠️  Не удалось определить ориентацию: {file_path.name}")
                    continue

                alias = self.get_alias(orientation)

                if self.has_alias_prefix(file_path.name):
                    processed_count += 1
                    continue

                new_filename = f"{alias}_{file_path.name}"
                new_file_path = file_path.parent / new_filename

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
    parser = argparse.ArgumentParser(description='Face Orientation Detector')
    parser.add_argument('directory', help='Путь к директории с изображениями')
    args = parser.parse_args()

    detector = FaceOrientationDetector()
    detector.process_directory(args.directory)


if __name__ == "__main__":
    main()