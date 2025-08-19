import warnings
from pytest import warns

# Фильтрация предупреждений от google.protobuf
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google._upb._message")

import unittest
from unittest.mock import Mock, patch
from PIL import Image
import numpy as np
from pathlib import Path

from src.infrastructure.detection.yolo_detector import YOLOFaceDetector
from src.domain.face import BoundingBox
import os

def test_detection_empty():
    detector = YOLOFaceDetector()

    # Загружаем изображение
    image_path = os.path.abspath(str("./tests/images/group2/watermelon.jpg"))
    image = Image.open(image_path)  # Используем PIL для загрузки

    # Perform detection
    result = detector.detect(image)  # Передаем само изображение, а не путь

    # Should be empty for watermelon image
    assert result == []  # Исправлен assert

def test_detection_not_empty():
    detector = YOLOFaceDetector()

    # Загружаем изображение
    image_path = os.path.abspath(str("/Users/user/Downloads/5dwntg0.jpeg"))
    image = Image.open(image_path)  # Используем PIL для загрузки

    # Perform detection
    result = detector.detect(image)

    print(result)

    # Проверяем количество найденных лиц
    assert len(result) == 1