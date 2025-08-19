# tests/unit/domain/test_bounding_box.py
from src.domain.face import BoundingBox


def test_bounding_box_properties():
    """Проверяет свойства bounding box"""
    bbox = BoundingBox(x1=10, y1=20, x2=50, y2=60)

    assert bbox.width == 40
    assert bbox.height == 40
    assert bbox.area() == 1600


def test_bounding_box_to_list():
    """Проверяет преобразование bounding box в список"""
    bbox = BoundingBox(x1=10, y1=20, x2=50, y2=60)
    assert bbox.to_list() == [10, 20, 50, 60]


def test_bounding_box_contains_point():
    """Проверяет проверку содержания точки в bounding box"""
    bbox = BoundingBox(x1=10, y1=20, x2=50, y2=60)

    # Точка внутри
    assert bbox.contains_point(20, 30) is True

    # Точка на границе
    assert bbox.contains_point(10, 20) is True
    assert bbox.contains_point(50, 60) is True

    # Точка снаружи
    assert bbox.contains_point(5, 15) is False
    assert bbox.contains_point(60, 70) is False