import pytest

from src.domain.face import BoundingBox
from src.infrastructure.detection.yolo_detector import \
    DefaultBoundingBoxProcessor


@pytest.fixture
def bbox_processor():
    return DefaultBoundingBoxProcessor()


def test_calculate_square_crop(bbox_processor):
    """Проверяет расчет квадратного обрезания"""
    bbox = BoundingBox(x1=10, y1=20, x2=50, y2=60)
    image_size = (100, 100)

    square_bbox = bbox_processor.calculate_square_crop(bbox, image_size)

    assert square_bbox.x1 >= 0
    assert square_bbox.y1 >= 0
    assert square_bbox.width == square_bbox.height


def test_merge_overlapping(bbox_processor):
    """Проверяет объединение пересекающихся bounding box'ов"""
    boxes = [
        BoundingBox(x1=10, y1=10, x2=30, y2=30),
        BoundingBox(x1=20, y1=20, x2=40, y2=40)
    ]

    merged_boxes = bbox_processor.merge_overlapping(boxes)

    assert len(merged_boxes) == 1
    assert merged_boxes[0].x1 <= 10
    assert merged_boxes[0].y1 <= 10
    assert merged_boxes[0].x2 >= 40
    assert merged_boxes[0].y2 >= 40


def test_merge_overlapping_no_overlap(bbox_processor):
    """Проверяет обработку bounding box'ов без пересечения"""
    boxes = [
        BoundingBox(x1=10, y1=10, x2=30, y2=30),
        BoundingBox(x1=50, y1=50, x2=80, y2=80)
    ]

    merged_boxes = bbox_processor.merge_overlapping(boxes)

    assert len(merged_boxes) == 2
    # Проверяем, что bounding boxes остались теми же
    assert merged_boxes[0].x1 == 10 and merged_boxes[0].y1 == 10
    assert merged_boxes[1].x1 == 50 and merged_boxes[1].y1 == 50