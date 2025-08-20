"""
Модуль тестов для YOLO детектора лиц.
"""

import unittest
import os
from PIL import Image  # type: ignore
from src.infrastructure.detection.yolo_detector import YOLOFaceDetector


class TestYOLOFaceDetector(unittest.TestCase):
    """Тесты для YOLOFaceDetector"""

    def test_detect_faces(self) -> None:
        """Тест обнаружения лиц"""

        image_path = os.path.abspath(str("/Users/user/Downloads/5dwntg0.jpeg"))
        image = Image.open(image_path)  # Используем PIL для загрузки

        detector = YOLOFaceDetector()
        results = detector.detect(image)

        self.assertEqual(len(results), 1)

        # проверка на нулевую длину
        self.assertNotEqual(results[0][0], results[0][2])
        self.assertNotEqual(results[0][1], results[0][3])

        self.assertGreaterEqual(results[0][0], 0)
        self.assertGreaterEqual(results[0][1], 0)
        self.assertGreater(results[0][2], 0)
        self.assertGreater(results[0][3], 0)



    def test_empty_detection(self) -> None:
        """Тест пустого результата обнаружение"""

        image_path = os.path.abspath(str("./tests/images/group2/watermelon.jpg"))
        image = Image.open(image_path)  # Используем PIL для загрузки

        detector = YOLOFaceDetector()
        results = detector.detect(image)

        self.assertEqual(len(results), 0)
