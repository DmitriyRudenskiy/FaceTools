# tests/unit/infrastructure/detection/test_yolo_detector.py
import pytest
from unittest.mock import MagicMock, patch
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
    mock_results.boxes.xyxy = [[10, 10, 50, 50], [60, 60, 90, 90]]
    mock_model.return_value = [mock_results]

    # Создаем тестовое изображение
    mock_image = MagicMock()

    # Вызываем метод
    boxes = yolo_detector.detect(mock_image)

    # Проверяем результат
    assert len(boxes) == 2
    assert boxes[0] == [10, 10, 50, 50]
    assert boxes[1] == [60, 60, 90, 90]


def test_model_loading_success():
    """Проверяет успешную загрузку модели"""
    with patch('os.path.exists') as mock_exists, \
            patch('ultralytics.YOLO') as mock_yolo:
        mock_exists.return_value = True
        mock_yolo.return_value = MagicMock()  # Добавляем возвращаемое значение

        detector = YOLOFaceDetector()

        assert detector.model is not None
        mock_yolo.assert_called_once()  # Используем assert_called_once вместо assert_called


def test_model_loading_failure():
    """Проверяет обработку ошибки загрузки модели"""
    with patch('os.path.exists') as mock_exists, \
            patch('ultralytics.YOLO') as mock_yolo:
        mock_exists.return_value = False
        mock_yolo.side_effect = Exception("Model load failed")

        with pytest.raises(Exception):
            YOLOFaceDetector()