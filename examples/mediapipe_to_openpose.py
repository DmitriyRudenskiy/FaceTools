import argparse
import json
import logging
import os
from glob import glob
from os.path import join, splitext

import cv2
import mediapipe as mp
import numpy as np
from google.protobuf.json_format import MessageToDict
from natsort import natsorted

# Конфигурация логгера
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description='Mediapipe to OpenPose JSON Converter')
    parser.add_argument('--input_folder', required=True, help='Path to images folder')
    parser.add_argument('--output_folder', default=None, help='Output folder for JSON (default: same as input)')
    parser.add_argument('--write_json', action='store_true', help='Enable JSON export')
    parser.add_argument('--model_complexity', type=int, default=2, choices=[0, 1, 2], help='Mediapipe model complexity')
    parser.add_argument('--min_detection_confidence', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--min_tracking_confidence', type=float, default=0.75, help='Tracking confidence threshold')
    args = parser.parse_args()

    # Если выходная папка не указана, используем входную
    output_folder = args.output_folder or args.input_folder

    # Инициализация Mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_drawing_styles = mp.solutions.drawing_styles

    # Шаблон OpenPose JSON
    json_template = {
        "version": 1.3,
        "people": [{
            "person_id": [-1],
            "pose_keypoints_2d": [],
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": []
        }]
    }

    # Поиск изображений
    img_extensions = ('*.png', '*.jpg', '*.jpeg')
    img_files = []
    for ext in img_extensions:
        img_files.extend(glob(join(args.input_folder, ext)))

    if not img_files:
        log.warning(f"No images found in {args.input_folder} with extensions {img_extensions}")
        return

    with mp_pose.Pose(
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            model_complexity=args.model_complexity,
            smooth_landmarks=True
    ) as pose:

        for img_path in natsorted(img_files):
            log.info(f"Processing: {img_path}")
            frame = cv2.imread(img_path)
            if frame is None:
                log.error(f"Failed to read image: {img_path}")
                continue

            height, width, _ = frame.shape
            results = pose.process(frame)

            if not results.pose_landmarks:
                log.warning(f"No pose detected in {img_path}")
                continue

            # Визуализация и сохранение
            annotated_frame = frame.copy()
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            base_name = os.path.basename(img_path)
            img_out_path = join(output_folder, f"{splitext(base_name)[0]}_keypoints{splitext(base_name)[1]}")
            cv2.imwrite(img_out_path, annotated_frame)
            log.info(f"Saved annotated image: {img_out_path}")

            if args.write_json:
                landmarks = results.pose_landmarks.landmark
                tmp = []
                for landmark in landmarks:
                    tmp.append((
                        landmark.x,
                        landmark.y,
                        landmark.visibility
                    ))

                # Масштабирование координат
                scaled = []
                for x, y, vis in tmp:
                    scaled.append((
                        int(x * width),
                        int(y * height),
                        vis
                    ))

                # Расчет дополнительных точек
                neck = (
                    (scaled[11][0] + scaled[12][0]) // 2,
                    (scaled[11][1] + scaled[12][1]) // 2,
                    0.95
                )
                mid_hip = (
                    (scaled[23][0] + scaled[24][0]) // 2,
                    (scaled[23][1] + scaled[24][1]) // 2,
                    0.95
                )

                # Порядок ключевых точек OpenPose (25 точек)
                op_indices = [
                    0,  # Нос
                    1,  # Шея (заменяется)
                    16,  # Правый глаз
                    15,  # Левый глаз
                    18,  # Правое ухо
                    17,  # Левое ухо
                    5,  # Левое плечо
                    2,  # Правый локоть
                    8,  # Левая кисть
                    6,  # Правое плечо
                    3,  # Левый локоть
                    7,  # Правая кисть
                    12,  # Левое бедро
                    9,  # Левое колено
                    10,  # Левая лодыжка
                    13,  # Правое бедро
                    14,  # Правое колено
                    11,  # Правая лодыжка
                    24,  # Левая ступня
                    22,  # Левая пятка
                    23,  # Правая ступня
                    21,  # Правая пятка
                    19,  # Левая кисть (мизинец)
                    20,  # Левая кисть (запястье)
                    18  # Правая кисть (мизинец)
                ]

                # Формирование точек с заменой специальных
                op_points = [scaled[i] for i in op_indices]
                op_points[1] = neck  # Шея
                op_points[12] = mid_hip  # Центр бедер

                # Преобразование в плоский список
                flat_list = []
                for point in op_points:
                    flat_list.extend(point)

                # Формирование JSON
                json_data = json_template.copy()
                json_data["people"][0]["pose_keypoints_2d"] = flat_list

                # Сохранение JSON
                json_path = join(output_folder, f"{splitext(base_name)[0]}_keypoints.json")
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                log.info(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()