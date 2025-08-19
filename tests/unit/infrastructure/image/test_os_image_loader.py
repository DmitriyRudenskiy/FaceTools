# tests/unit/infrastructure/image/test_os_image_loader.py
import pytest
from unittest.mock import MagicMock, patch
from src.infrastructure.image.os_image_loader import OSImageLoader
from src.domain.image_model import Image


@pytest.fixture
def image_loader():
    return OSImageLoader()


def test_load_success(image_loader):
    """Проверяет успешную загрузку изображения"""
    with patch('PIL.Image.open') as mock_open, \
            patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        mock_image = MagicMock()
        mock_open.return_value = mock_image

        # Загружаем изображение
        result = image_loader.load("/test/image.jpg")

        # Проверяем результат
        assert isinstance(result, Image)
        assert result.data == mock_image
        assert result.info is not None
        mock_open.assert_called_once_with("/test/image.jpg")


def test_load_failure(image_loader):
    """Проверяет обработку ошибки загрузки изображения"""
    with patch('os.path.exists') as mock_exists, \
            patch('PIL.Image.open') as mock_open:
        mock_exists.return_value = True
        mock_open.side_effect = Exception("Image load failed")

        with pytest.raises(Exception):
            image_loader.load("/test/image.jpg")


def test_load_images_from_directory(image_loader, tmp_path):
    """Проверяет загрузку изображений из директории"""
    # Создаем временную директорию с тестовыми изображениями
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "image1.jpg").touch()
    (test_dir / "image2.png").touch()

    # Загружаем изображения
    images = image_loader.load_images(str(test_dir))

    # Проверяем результат
    assert len(images) == 2
    assert all(isinstance(img[0], str) for img in images)  # Пути
    assert all(img[1] is not None for img in images)  # Изображения
    assert any("image1.jpg" in img[0] for img in images)
    assert any("image2.png" in img[0] for img in images)