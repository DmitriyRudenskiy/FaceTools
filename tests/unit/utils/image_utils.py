"""Unit tests for SquareCropCalculator."""

import unittest
from src.utils.image_utils import SquareCropCalculator


class TestSquareCropCalculator(unittest.TestCase):
    """Test cases for SquareCropCalculator functionality."""

    def test_initialization(self) -> None:
        """Test initialization with different padding ratios."""
        # Test 1: Initialization with padding_ratio=0.0
        calculator = SquareCropCalculator(padding_ratio=0.0)
        self.assertEqual(calculator.padding_ratio, 0.0)

        # Test 2: Initialization with padding_ratio=0.2
        calculator = SquareCropCalculator(padding_ratio=0.2)
        self.assertEqual(calculator.padding_ratio, 0.2)

    def test_no_padding_square_bbox(self) -> None:
        """Test crop calculation for square bounding box without padding."""
        calculator = SquareCropCalculator(padding_ratio=0.0)
        bbox = (10, 10, 20, 20)
        crop = calculator.calculate_crop(bbox, 100, 100)
        self.assertEqual(crop, (10, 10, 20, 20))

    def test_no_padding_horizontal_rectangle(self) -> None:
        """Test crop calculation for horizontal rectangle without padding."""
        calculator = SquareCropCalculator(padding_ratio=0.0)
        bbox = (10, 10, 30, 20)
        crop = calculator.calculate_crop(bbox, 100, 100)
        # Center (20,15), side 20 -> (10,5,30,25)
        self.assertEqual(crop, (10, 5, 30, 25))

    def test_no_padding_vertical_rectangle(self) -> None:
        """Test crop calculation for vertical rectangle without padding."""
        calculator = SquareCropCalculator(padding_ratio=0.0)
        bbox = (10, 10, 20, 40)
        crop = calculator.calculate_crop(bbox, 40, 50)
        # Center (15,25), side 30 -> (0,10,30,40)
        self.assertEqual(crop, (0, 10, 30, 40))

    def test_with_padding(self) -> None:
        """Test crop calculation with padding."""
        calculator = SquareCropCalculator(padding_ratio=0.2)
        bbox = (10, 10, 20, 20)
        crop = calculator.calculate_crop(bbox, 100, 100)
        # Side 12, center (15,15) -> (9,9,21,21)
        self.assertEqual(crop, (9, 9, 21, 21))

    def test_near_left_edge(self) -> None:
        """Test crop near left image edge."""
        calculator = SquareCropCalculator(padding_ratio=0.0)
        bbox = (0, 10, 10, 20)
        crop = calculator.calculate_crop(bbox, 100, 100)
        # Center (5,15), side 10 -> (0,10,10,20)
        self.assertEqual(crop, (0, 10, 10, 20))

    def test_near_top_edge(self) -> None:
        """Test crop near top image edge."""
        calculator = SquareCropCalculator(padding_ratio=0.0)
        bbox = (10, 0, 20, 10)
        crop = calculator.calculate_crop(bbox, 100, 100)
        self.assertEqual(crop, (10, 0, 20, 10))

    def test_near_right_edge(self) -> None:
        """Test crop near right image edge."""
        calculator = SquareCropCalculator(padding_ratio=0.0)
        bbox = (90, 10, 100, 20)
        crop = calculator.calculate_crop(bbox, 100, 100)
        # Center (95,15), side 10 -> (90,10,100,20)
        self.assertEqual(crop, (90, 10, 100, 20))

    def test_near_bottom_edge(self) -> None:
        """Test crop near bottom image edge."""
        calculator = SquareCropCalculator(padding_ratio=0.0)
        bbox = (10, 90, 20, 100)
        crop = calculator.calculate_crop(bbox, 100, 100)
        self.assertEqual(crop, (10, 90, 20, 100))

    def test_crop_exceeds_image(self) -> None:
        """Test crop that exceeds image boundaries."""
        calculator = SquareCropCalculator(padding_ratio=2.0)
        bbox = (10, 10, 20, 20)
        crop = calculator.calculate_crop(bbox, 25, 25)
        # Side 30 -> cropped to (0,0,25,25)
        self.assertEqual(crop, (0, 0, 25, 25))

    def test_bbox_equals_image(self) -> None:
        """Test crop when bbox matches image size."""
        calculator = SquareCropCalculator(padding_ratio=0.0)
        bbox = (0, 0, 100, 100)
        crop = calculator.calculate_crop(bbox, 100, 100)
        self.assertEqual(crop, (0, 0, 100, 100))

    def test_integer_coordinates(self) -> None:
        """Test that crop coordinates are integers."""
        calculator = SquareCropCalculator(padding_ratio=0.1)
        bbox = (10, 10, 20, 20)
        crop = calculator.calculate_crop(bbox, 100, 100)
        self.assertTrue(all(isinstance(coord, int) for coord in crop))

    def test_zero_area_bbox(self) -> None:
        """Test crop with zero area bounding box."""
        calculator = SquareCropCalculator(padding_ratio=0.0)
        bbox = (10, 10, 10, 10)
        crop = calculator.calculate_crop(bbox, 100, 100)
        # Center (10,10), side 0 -> (10,10,10,10)
        self.assertEqual(crop, (10, 10, 10, 10))

    def test_large_padding(self) -> None:
        """Test crop with very large padding ratio."""
        calculator = SquareCropCalculator(padding_ratio=10.0)
        bbox = (10, 10, 20, 20)
        crop = calculator.calculate_crop(bbox, 100, 100)
        # Side 110 -> half_side=55, center (15,15) -> (0,0,70,70)
        self.assertEqual(crop, (0, 0, 70, 70))

    def test_large_padding_center_bbox(self) -> None:
        """Test large padding with centered bbox."""
        calculator = SquareCropCalculator(padding_ratio=10.0)
        bbox = (40, 40, 60, 60)  # Center (50,50)
        crop = calculator.calculate_crop(bbox, 100, 100)
        # Side 220 -> half_side=110 -> cropped to (0,0,100,100)
        self.assertEqual(crop, (0, 0, 100, 100))

    def test_padding_precision(self) -> None:
        """Test crop with fractional padding ratio."""
        calculator = SquareCropCalculator(padding_ratio=0.09)
        bbox = (10, 10, 20, 20)
        crop = calculator.calculate_crop(bbox, 100, 100)
        # Side 10*1.09=10.9 -> 10 -> (10,10,20,20)
        self.assertEqual(crop, (10, 10, 20, 20))

    def test_odd_side_length(self) -> None:
        """Test crop with odd side length bbox."""
        calculator = SquareCropCalculator(padding_ratio=0.0)
        bbox = (10, 10, 21, 21)  # Side 11
        crop = calculator.calculate_crop(bbox, 100, 100)
        # Center (15,15), half_side=5 -> (10,10,20,20)
        self.assertEqual(crop, (10, 10, 20, 20))


if __name__ == '__main__':
    unittest.main()
