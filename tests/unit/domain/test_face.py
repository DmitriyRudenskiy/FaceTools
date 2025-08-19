from src.domain.face import BoundingBox, Face
from src.domain.image_model import (Image,  # Добавлен импорт ImageInfo
                                    ImageInfo)


def test_face_initialization():
    """Проверяет инициализацию объекта Face"""
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
    assert "/test/image.jpg#10,20" in face_id