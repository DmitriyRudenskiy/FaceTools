from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image as PILImage

from src.application.services.face_crop_service import FaceCropService
from src.domain.face import BoundingBox
from src.domain.image_model import (Image,  # Добавлен импорт ImageInfo
                                    ImageInfo)


@pytest.fixture
def mock_dependencies():
    """Создает моки для всех зависимостей FaceCropService"""
    return {
        'file_organizer': MagicMock(),
        'face_detector': MagicMock(),
        'bbox_processor': MagicMock(),
        'image_loader': MagicMock()
    }


@pytest.fixture
def face_crop_service(mock_dependencies):
    return FaceCropService(
        file_organizer=mock_dependencies['file_organizer'],
        face_detector=mock_dependencies['face_detector'],
        bbox_processor=mock_dependencies['bbox_processor'],
        image_loader=mock_dependencies['image_loader']
    )


def create_test_image(size=(100, 100)):
    """Создает реальное тестовое изображение"""
    return PILImage.new('RGB', size, color='red')


def test_process_images_success(face_crop_service, mock_dependencies):
    """Проверяет успешную обработку изображений"""
    # Настройка моков
    mock_dependencies['file_organizer'].exists.return_value = True
    mock_dependencies['file_organizer'].is_directory.return_value = True

    # Создаем реальное изображение
    test_image = create_test_image()
    mock_dependencies['image_loader'].load.return_value = Image(
        data=np.array(test_image),
        info=ImageInfo(
            path="/test/image.jpg",
            size=(100, 100),
            format="jpg"
        )
    )

    # Создаем тестовые bounding boxes как объекты BoundingBox
    test_boxes = [
        BoundingBox(x1=10, y1=10, x2=50, y2=50),
        BoundingBox(x1=60, y1=60, x2=90, y2=90)
    ]

    # Мокаем детекцию лиц
    mock_dependencies['face_detector'].detect.return_value = test_boxes

    # Мокаем обработку bounding boxes
    mock_dependencies['bbox_processor'].merge_overlapping.return_value = test_boxes

    # Мокаем сохранение
    mock_dependencies['file_organizer'].save.return_value = "/output/test.jpg"

    # Вызываем метод
    result = face_crop_service.process_images("/test/input", "/test/output")

    # Проверяем результат
    assert result is True

    # Проверяем, что были вызваны необходимые методы
    mock_dependencies['image_loader'].load.assert_called()
    mock_dependencies['face_detector'].detect.assert_called()
    mock_dependencies['bbox_processor'].merge_overlapping.assert_called()
    assert mock_dependencies['file_organizer'].save.call_count > 0


def test_process_images_no_faces_detected(face_crop_service, mock_dependencies):
    """Проверяет обработку случая, когда лица не обнаружены"""
    mock_dependencies['file_organizer'].exists.return_value = True
    mock_dependencies['file_organizer'].is_directory.return_value = True

    # Создаем реальное изображение
    test_image = create_test_image()
    mock_dependencies['image_loader'].load.return_value = Image(
        data=np.array(test_image),
        info=ImageInfo(
            path="/test/image.jpg",
            size=(100, 100),
            format="jpg"
        )
    )

    # Детектор возвращает пустой список
    mock_dependencies['face_detector'].detect.return_value = []

    result = face_crop_service.process_images("/test/input", "/test/output")
    assert result is True  # Должно вернуть True, даже если лиц нет


def test_detect_faces(face_crop_service, mock_dependencies):
    """Проверяет детекцию лиц"""
    # Настройка моков
    test_image = create_test_image()
    mock_dependencies['image_loader'].load.return_value = Image(
        data=np.array(test_image),
        info=ImageInfo(
            path="/test/image.jpg",
            size=(100, 100),
            format="jpg"
        )
    )

    # Создаем тестовые bounding boxes как объекты BoundingBox
    test_boxes = [
        BoundingBox(x1=10, y1=10, x2=50, y2=50),
        BoundingBox(x1=60, y1=60, x2=90, y2=90)
    ]

    # Мокаем детекцию лиц
    mock_dependencies['face_detector'].detect.return_value = test_boxes

    # Мокаем обработку bounding boxes
    mock_dependencies['bbox_processor'].merge_overlapping.return_value = test_boxes

    # Вызываем метод
    result = face_crop_service.process_images("/test/input", "/test/output")

    # Проверяем результат
    assert result is True
    assert len(test_boxes) == 2