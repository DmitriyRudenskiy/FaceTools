# tests/unit/application/services/test_face_crop_service.py
import pytest
from unittest.mock import MagicMock
from src.application.services.face_crop_service import FaceCropService
from src.domain.face import Face, BoundingBox
from src.domain.image_model import Image


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
    """Создает экземпляр FaceCropService с мокнутыми зависимостями"""
    return FaceCropService(**mock_dependencies)


def test_process_images_success(face_crop_service, mock_dependencies):
    """Проверяет успешную обработку изображений"""
    # Настройка моков
    mock_dependencies['file_organizer'].exists.return_value = True
    mock_dependencies['image_loader'].load.return_value = Image(
        data=MagicMock(),
        info=MagicMock(size=(100, 100))
    )

    # Создаем тестовые bounding boxes
    test_boxes = [
        [10, 10, 50, 50],
        [60, 60, 90, 90]
    ]

    # Мокаем детекцию лиц
    mock_dependencies['face_detector'].detect.return_value = test_boxes

    # Мокаем обработку bounding boxes
    mock_dependencies['bbox_processor'].merge_overlapping.return_value = test_boxes

    # Вызываем метод
    result = face_crop_service.process_images("/test/input", "/test/output")

    # Проверяем результат
    assert result is True
    # Проверяем, что были вызваны необходимые методы
    mock_dependencies['image_loader'].load.assert_called()
    mock_dependencies['face_detector'].detect.assert_called()
    mock_dependencies['bbox_processor'].merge_overlapping.assert_called()
    mock_dependencies['file_organizer'].save.assert_called()


def test_process_images_directory_not_exists(face_crop_service, mock_dependencies):
    """Проверяет обработку несуществующей директории"""
    mock_dependencies['file_organizer'].exists.return_value = False

    result = face_crop_service.process_images("/non/existent/dir", "/test/output")
    assert result is False


def test_process_images_no_faces_detected(face_crop_service, mock_dependencies):
    """Проверяет обработку случая, когда лица не обнаружены"""
    mock_dependencies['file_organizer'].exists.return_value = True
    mock_dependencies['image_loader'].load.return_value = Image(
        data=MagicMock(),
        info=MagicMock(size=(100, 100))
    )
    mock_dependencies['face_detector'].detect.return_value = []

    result = face_crop_service.process_images("/test/input", "/test/output")
    assert result is True  # Должно вернуть True, даже если лиц нет
    assert mock_dependencies['file_organizer'].save.call_count == 0


def test_detect_faces(face_crop_service, mock_dependencies):
    """Проверяет детекцию лиц"""
    # Настройка моков
    mock_image = Image(data=MagicMock(), info=MagicMock(size=(100, 100)))
    mock_dependencies['face_detector'].detect.return_value = [[10]][[10]][[50]][[50]]
    mock_dependencies['bbox_processor'].merge_overlapping.return_value = [[10]][[10]][[50]][[50]]

    # Вызываем метод
    faces = face_crop_service._detect_faces(mock_image)

    # Проверяем результат
    assert len(faces) == 1
    assert isinstance(faces[0], Face)
    assert isinstance(faces[0].bounding_box, BoundingBox)
    assert faces[0].bounding_box.x1 == 10
    assert faces[0].bounding_box.y1 == 10
    assert faces[0].bounding_box.x2 == 50
    assert faces[0].bounding_box.y2 == 50
    assert faces[0].landmarks is None
    assert faces[0].embedding == []
    assert faces[0].image == mock_image


def test_crop_face(face_crop_service, mock_dependencies):
    """Проверяет обрезку лица по bounding box"""
    # Создаем тестовые объекты
    mock_image = Image(data=MagicMock(), info=MagicMock(size=(100, 100)))
    bbox = BoundingBox(x1=10, y1=10, x2=50, y2=50)

    # Мокаем обработку bounding box
    mock_dependencies['bbox_processor'].calculate_square_crop.return_value = (10, 10, 50, 50)

    # Вызываем метод
    cropped_image = face_crop_service._crop_face(mock_image, bbox)

    # Проверяем результат
    mock_dependencies['bbox_processor'].calculate_square_crop.assert_called_once_with(
        bbox, (100, 100)
    )
    assert cropped_image is not None


def test_generate_output_path(face_crop_service, mock_dependencies):
    """Проверяет генерацию пути для сохранения лица"""
    mock_dependencies['file_organizer'].get_basename.return_value = "test_image"

    # Вызываем метод
    output_path = face_crop_service._generate_output_path(
        "/path/to/test_image.jpg", 0, "/output/dir"
    )

    # Проверяем результат
    assert output_path == "/output/dir/test_image_face_1.jpg"

    # Проверяем для второго лица
    output_path = face_crop_service._generate_output_path(
        "/path/to/test_image.jpg", 1, "/output/dir"
    )
    assert output_path == "/output/dir/test_image_face_2.jpg"