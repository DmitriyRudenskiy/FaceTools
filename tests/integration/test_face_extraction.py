# tests/integration/test_face_extraction.py
import os
from unittest.mock import MagicMock, patch
import tempfile
import shutil
import pytest
from unittest.mock import patch
from src.config.dependency_injector import DependencyInjector
from src.application.services.face_crop_service import FaceCropService
from src.domain.face import BoundingBox


@pytest.fixture
def test_image_path(tmp_path):
    """Создает тестовое изображение"""
    # Здесь можно создать простое изображение для тестов
    # Для реального теста нужно использовать библиотеку PIL для создания изображения
    test_image = tmp_path / "test.jpg"
    # В реальном тесте здесь будет код для создания тестового изображения
    test_image.touch()
    return str(test_image)


@pytest.fixture
def test_input_dir(tmp_path, test_image_path):
    """Создает тестовую директорию с изображениями"""
    # Копируем тестовое изображение в директорию
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(test_image_path, input_dir / "test.jpg")
    return str(input_dir)


def test_face_extraction_end_to_end(tmp_path):
    """Проверяет полный процесс извлечения лиц от начала до конца"""
    # Создаем тестовую структуру
    test_dir = tmp_path / "input"
    test_dir.mkdir()
    (test_dir / "test.jpg").touch()

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
def test_face_extraction_with_mocked_detector(mock_yolo, test_input_dir):
    """Проверяет извлечение лиц с мокнутым детектором"""
    # Настраиваем мок детектора
    mock_results = MagicMock()
    mock_results.boxes.xyxy = [BoundingBox(10, 10, 50, 50)]
    mock_yolo.return_value = [mock_results]

    # Создаем сервис
    injector = DependencyInjector()
    service = injector.get_face_crop_service()

    # Обрабатываем изображения
    with tempfile.TemporaryDirectory() as output_dir:
        result = service.process_images(test_input_dir, output_dir)
        assert result is True
        face_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
        assert len(face_files) == 1  # Ожидаем одно лицо