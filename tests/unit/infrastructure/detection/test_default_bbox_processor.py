# tests/unit/infrastructure/detection/test_default_bbox_processor.py
import pytest
from src.infrastructure.detection.yolo_detector import DefaultBoundingBoxProcessor
from src.domain.face import BoundingBox


@pytest.fixture
def bbox_processor():
    return DefaultBoundingBoxProcessor()


def test_calculate_square_crop(bbox_processor):
    """Проверяет вычисление квадратного обрезка"""
    # Тестовый bounding box
    bbox = BoundingBox(x1=10, y1=20, x2=50, y2=60)
    image_size = (100, 100)

    # Вычисляем квадратный обрезок
    crop_coords = bbox_processor.calculate_square_crop(bbox, image_size)

    # Проверяем результат
    assert len(crop_coords) == 4
    assert all(isinstance(coord, int) for coord in crop_coords)
    assert crop_coords[0] >= 0  # x1
    assert crop_coords[1] >= 0  # y1
    assert crop_coords[2] <= 100  # x2
    assert crop_coords[3] <= 100  # y2
    assert crop_coords[2] - crop_coords[0] == crop_coords[3] - crop_coords[1]  # квадрат


def test_merge_overlapping(bbox_processor):
    """Проверяет объединение пересекающихся bounding box'ов"""
    # Создаем несколько пересекающихся bounding box'ов
    boxes = [
        BoundingBox(10, 10, 50, 50),
        BoundingBox(15, 15, 55, 55),
        BoundingBox(60, 60, 90, 90)
    ]

    # Объединяем
    merged_boxes = bbox_processor.merge_overlapping(boxes)

    # Проверяем результат
    assert len(merged_boxes) == 2  # Должно быть 2 непересекающихся box'а
    assert merged_boxes[0] == BoundingBox(10, 10, 55, 55)  # Объединенный box
    assert merged_boxes[1] == BoundingBox(60, 60, 90, 90)  # Отдельный box


def test_merge_overlapping_no_overlap(bbox_processor):
    """Проверяет обработку bounding box'ов без пересечения"""
    boxes = [
        [10, 10, 30, 30],
        [50, 50, 80, 80]
    ]

    merged_boxes = bbox_processor.merge_overlapping(boxes)
    assert len(merged_boxes) == 2
    assert merged_boxes == boxes