from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

from src.infrastructure.detection.yolo_detector import YOLOFaceDetector


@pytest.fixture
def yolo_detector():
    with patch('ultralytics.YOLO'):
        return YOLOFaceDetector()


def test_detect_faces(yolo_detector):
    """Проверяет детекцию лиц"""
    # Мокаем модель и результаты
    mock_model = MagicMock()
    yolo_detector.model = mock_model

    # Создаем тестовые данные
    mock_results = MagicMock()
    mock_results.boxes.xyxy = torch.tensor([[10, 10, 50, 50], [60, 60, 90, 90]])
    mock_model.return_value = [mock_results]

    # Создаем тестовое изображение
    mock_image = np.array(PILImage.new('RGB', (100, 100), color='red'))

    # Вызываем метод
    boxes = yolo_detector.detect(mock_image)

    # Проверяем результат
    assert len(boxes) == 2
    assert boxes[0].x1 == 10 and boxes[0].y1 == 10 and boxes[0].x2 == 50 and boxes[0].y2 == 50
    assert boxes[1].x1 == 60 and boxes[1].y1 == 60 and boxes[1].x2 == 90 and boxes[1].y2 == 90


def test_model_loading_success():
    """Проверяет успешную загрузку модели"""
    with patch('os.path.exists') as mock_exists, \
            patch('ultralytics.YOLO') as mock_yolo:
        mock_exists.return_value = True
        mock_yolo.return_value = MagicMock()

        detector = YOLOFaceDetector()

        assert detector.model is not None
        mock_yolo.assert_called()  # Проверяем, что YOLO был вызван


def test_model_loading_failure():
    """Проверяет обработку ошибки загрузки модели"""
    with patch('os.path.exists') as mock_exists, \
            patch('ultralytics.YOLO') as mock_yolo:
        mock_exists.return_value = False
        mock_yolo.side_effect = Exception("Model load failed")

        with pytest.raises(Exception):
            YOLOFaceDetector()