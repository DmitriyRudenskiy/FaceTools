"""Utility functions for image processing operations, including face cropping."""

from typing import Tuple

class SquareCropCalculator:
    """Class for calculating square crops centered around detected faces."""

    def __init__(self, padding_ratio: float = 0.0):
        """
        Initialize the square crop calculator.

        Args:
            padding_ratio: Ratio of padding to add around the face (0.0 for no padding)
        """
        self.padding_ratio = padding_ratio

    def calculate_crop(
        self, bbox: Tuple[int, int, int, int], image_width: int, image_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Calculate coordinates for a square crop centered on the face with padding.

        Args:
            bbox: Tuple of (x1, y1, x2, y2) coordinates of the face bounding box
            image_width: Width of the original image
            image_height: Height of the original image

        Returns:
            Tuple of (crop_x1, crop_y1, crop_x2, crop_y2) for the crop area
        """
        x1, y1, x2, y2 = bbox

        # Calculate center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Calculate base side length
        side_length = max(x2 - x1, y2 - y1)

        # Apply padding if needed
        if self.padding_ratio > 0:
            side_length = int(side_length * (1 + self.padding_ratio))

        # Calculate half side length once
        half_side = side_length // 2

        # Calculate and return crop coordinates with boundary checks
        return (
            max(0, center_x - half_side),
            max(0, center_y - half_side),
            min(image_width, center_x + half_side),
            min(image_height, center_y + half_side)
        )
