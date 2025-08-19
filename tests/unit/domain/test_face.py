# tests/unit/domain/test_face.py
import pytest
from src.domain.face import Face, BoundingBox
from src.domain.image_model import Image, ImageInfo


def test_face_initialization():
    """Проверяет инициализацию объекта Face"""
    bbox = BoundingBox(x1=10, y1=20, x2=50, y2=60)
    image = Image(data=None, info=None)

    face = Face(
        bounding_box=bbox,
        landmarks=None,
        embedding=[0.1, 0.2, 0.3],
        image=image,
        is_reference=False,
        confidence=0.8
    )

    assert face.bounding_box == bbox
    assert face.landmarks is None
    assert face.embedding == [0.1, 0.2, 0.3]
    assert face.image == image
    assert face.is_reference is False
    assert face.confidence == 0.8


def test_face_id_generation():
    """Проверяет генерацию уникального идентификатора лица"""
    bbox = BoundingBox(x1=10, y1=20, x2=50, y2=60)
    image = Image(
        data=None,
        info=ImageInfo(
            path="/test/image.jpg",
            size=(100, 100),
            format="jpg"
        )
    )

    face = Face(
        bounding_box=bbox,
        landmarks=None,
        embedding=[],
        image=image
    )

    # Проверяем, что face_id содержит путь к изображению и координаты
    face_id = face.face_id
    assert "/test/image.jpg" in face_id
    assert "10,20" in face_id  # округленные координаты


def test_face_confidence_validation():
    """Проверяет валидацию confidence"""
    bbox = BoundingBox(x1=10, y1=20, x2=50, y2=60)
    image = Image(data=None, info=None)

    # Проверяем, что confidence вне диапазона [0,1] корректируется
    face_high = Face(
        bounding_box=bbox,
        landmarks=None,
        embedding=[],
        image=image,
        confidence=1.5
    )
    assert face_high.confidence == 1.0

    face_low = Face(
        bounding_box=bbox,
        landmarks=None,
        embedding=[],
        image=image,
        confidence=-0.5
    )
    assert face_low.confidence == 1.0