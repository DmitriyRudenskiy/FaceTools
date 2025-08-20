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
    parser.add_argument('-p', '--padding', type=float, default=0.0,
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
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_calculator.calculate_crop(
                    bbox, img_width, img_height
                )
                crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                output_path = Path(args.output) / f"{base_name}_face_{i + 1}.jpg"
                crop_img.save(output_path)
                print(f"Saved face: {output_path}")

if __name__ == "__main__":
    main()