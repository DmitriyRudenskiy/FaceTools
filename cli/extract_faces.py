"""Module for extracting faces from images using YOLO face detection."""

import argparse
import os
import sys
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

def main() -> None:
    """Main function to handle face extraction from images."""
    parser = argparse.ArgumentParser(description='Detect and save faces from images')
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

    for image_path in images:

        faces = detector.detect(image_path)
        base_name = Path(image_path).stem

        with Image.open(image_path) as img:
            img_width, img_height = img.size
            img = img.convert('RGB')

            for i, bbox in enumerate(faces):
                # Calculate target square coordinates without boundary constraints
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                face_width = x2 - x1
                face_height = y2 - y1
                base_side = max(face_width, face_height)

                # Calculate side length with padding
                side_length = base_side
                if crop_calculator.padding_ratio > 0:
                    side_length = int(base_side * (1 + crop_calculator.padding_ratio))

                half_side = side_length // 2
                target_x1 = center_x - half_side
                target_y1 = center_y - half_side
                target_x2 = center_x + half_side
                target_y2 = center_y + half_side

                # Get boundary-constrained crop coordinates
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_calculator.calculate_crop(
                    bbox, img_width, img_height
                )

                # Calculate offsets within the target square (ensure integers)
                x_offset = int(crop_x1 - target_x1)
                y_offset = int(crop_y1 - target_y1)
                crop_width = crop_x2 - crop_x1
                crop_height = crop_y2 - crop_y1

                # Create a new square canvas filled with acid green
                acid_green = (0, 255, 0)  # Acid green color (RGB)
                square_img = Image.new('RGB', (side_length, side_length), acid_green)

                # Paste the visible part of the image into the square
                if crop_width > 0 and crop_height > 0:
                    crop_region = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    square_img.paste(crop_region, (x_offset, y_offset))

                output_path = Path(args.output) / f"{base_name}_face_{i + 1}.jpg"
                square_img.save(output_path)
                print(f"Saved face: {output_path}")

if __name__ == "__main__":
    main()