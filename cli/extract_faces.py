"""Module for extracting faces from images using YOLO face detection with consolidated coordinates export."""

import argparse
import os
import sys
import json
from pathlib import Path
from PIL import Image

# Add project root to path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# pylint: disable=import-error,wrong-import-position
from src.infrastructure.file.image_loader import ImageLoader
from src.infrastructure.detection.yolo_detector import YOLOFaceDetector
from src.utils.image_utils import SquareCropCalculator
# pylint: enable=import-error,wrong-import-position

# Токсично зеленый цвет (#39FF14)
TOXIC_GREEN = (57, 255, 20)

def create_square_face_with_padding(cropped_face, target_size):
    """
    Создает квадратное изображение с заполнением недостающего места токсично зеленым цветом.
    Лицо располагается в верхней части, зеленое заполнение снизу.

    Args:
        cropped_face: PIL Image - обрезанное изображение лица
        target_size: int - целевой квадратный размер

    Returns:
        PIL Image - квадратное изображение с зеленой подложкой
    """
    # Создаем новое изображение с токсично зеленым фоном
    square_image = Image.new('RGB', (target_size, target_size), TOXIC_GREEN)

    # Вычисляем размеры лица
    face_width, face_height = cropped_face.size

    # Лицо располагается в верхней части
    x_offset = target_size - face_width
    y_offset = target_size - face_height

    # Вставляем лицо в верхнюю часть зеленого фона
    square_image.paste(cropped_face, (x_offset, y_offset))

    return square_image

def main() -> None:
    """Main function to handle face extraction from images."""
    parser = argparse.ArgumentParser(description='Detect and save faces from images with consolidated coordinates')
    parser.add_argument('-s', '--source', required=True, help='Path to image directory')
    parser.add_argument('-o', '--output', default='faces', help='Output directory')
    parser.add_argument('-p', '--padding', type=float, default=0.3,
                        help='Padding ratio around face (e.g., 0.2 for 20%%)')
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    loader = ImageLoader()
    detector = YOLOFaceDetector()
    crop_calculator = SquareCropCalculator(padding_ratio=args.padding)

    images = loader.load_images(args.source)
    all_faces_data = []  # Список для хранения всех данных о лицах

    for image_path in images:
        faces = detector.detect(image_path)
        base_name = Path(image_path).stem

        with Image.open(image_path) as img:
            img_width, img_height = img.size
            img = img.convert('RGB')

            for i, bbox in enumerate(faces):
                # Calculate crop coordinates and ensure integers
                crop_x1, crop_y1, crop_x2, crop_y2 = map(int, crop_calculator.calculate_crop(
                    bbox, img_width, img_height
                ))

                # Calculate output size (square)
                side_length = int(max(crop_x2 - crop_x1, crop_y2 - crop_y1))

                # Crop the face region
                face_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

                # Создаем квадратное изображение с токсично зеленым заполнением
                if face_img.size != (side_length, side_length):
                    # Если изображение не квадратное, добавляем зеленую подложку снизу
                    square_face_img = create_square_face_with_padding(face_img, side_length)
                else:
                    # Если изображение уже квадратное, используем как есть
                    square_face_img = face_img

                # Save face image
                output_path = Path(args.output) / f"{base_name}_face_{i + 1}.jpg"
                square_face_img.save(output_path)
                print(f"Saved face: {output_path}")

                # Collect face data (without padding_ratio and timestamp)
                face_data = {
                    "image_path": str(Path(image_path).resolve()),
                    "face_index": i,
                    "bbox_original": [int(x) for x in bbox],
                    "bbox_cropped": [crop_x1, crop_y1, crop_x2, crop_y2],
                    "image_size": [img_width, img_height],
                    "output_size": [side_length, side_length],
                    "face_path": str(output_path.resolve()),
                    "padding_color": "toxic_green",
                    "padding_color_hex": "#39FF14",
                    "padding_position": "bottom",
                    "face_position": "top"
                }
                all_faces_data.append(face_data)

    # Save all face data to a single JSON file
    json_path = Path(args.output) / "faces_coordinates.json"
    with open(json_path, 'w') as f:
        json.dump(all_faces_data, f, indent=2)

    print(f"Saved consolidated coordinates to: {json_path}")
    print(f"Total faces processed: {len(all_faces_data)}")

if __name__ == "__main__":
    main()