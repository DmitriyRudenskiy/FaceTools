from unittest.mock import MagicMock, patch

from PIL import Image as PILImage  # Добавлен импорт PIL

from src.config.dependency_injector import DependencyInjector


def create_test_image(file_path, size=(100, 100)):
    """Создает реальное тестовое изображение"""
    img = PILImage.new('RGB', size, color='red')
    img.save(file_path)


def test_face_extraction_end_to_end(tmp_path):
    """Проверяет полный процесс извлечения лиц от начала до конца"""
    # Создаем тестовую структуру
    test_dir = tmp_path / "input"
    test_dir.mkdir()

    # Создаем реальное изображение вместо пустого файла
    test_image = test_dir / "test.jpg"
    create_test_image(test_image)

    # Создаем сервис
    injector = DependencyInjector()
    service = injector.get_face_crop_service()

    # Создаем временную директорию для вывода
    output_dir = tmp_path / "output"

    # Обрабатываем изображения
    result = service.process_images(str(test_dir), str(output_dir))

    # Проверяем результат
    assert result is True

    # Проверяем, что в output_dir есть файлы с лицами
    face_files = list(output_dir.glob("*.jpg"))
    assert len(face_files) > 0


@patch('src.infrastructure.detection.yolo_detector.YOLO')
def test_face_extraction_with_mocked_detector(mock_yolo, tmp_path):
    """Проверяет извлечение лиц с мокнутым детектором"""
    # Создаем тестовую структуру
    test_dir = tmp_path / "input"
    test_dir.mkdir()

    # Создаем реальное изображение
    test_image = test_dir / "test.jpg"
    create_test_image(test_image)

    # Настраиваем мок детектора
    mock_results = MagicMock()
    # Используем правильный формат для bounding boxes
    mock_results.boxes.xyxy = [[10]][[10]][[50]][[50]]
    mock_yolo.return_value = [mock_results]

    # Создаем сервис
    injector = DependencyInjector()
    service = injector.get_face_crop_service()

    # Обрабатываем изображения
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = service.process_images(str(test_dir), str(output_dir))
    assert result is True

    # Проверяем, что файлы созданы
    face_files = list(output_dir.glob("*.jpg"))
    assert len(face_files) > 0