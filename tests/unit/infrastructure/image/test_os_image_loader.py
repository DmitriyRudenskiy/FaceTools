from unittest.mock import patch

import pytest
from PIL import Image as PILImage

from src.domain.image_model import Image, ImageInfo
from src.infrastructure.image.os_image_loader import OSImageLoader


@pytest.fixture
def image_loader():
    return OSImageLoader()


def test_load_success(image_loader):
    """Проверяет успешную загрузку изображения"""
    with patch('PIL.Image.open') as mock_open, \
            patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True

        # Создаем реальное изображение
        mock_image = PILImage.new('RGB', (100, 100), color='red')
        mock_open.return_value = mock_image

        # Загружаем изображение
        result = image_loader.load("/test/image.jpg")

        # Проверяем результат
        assert isinstance(result, Image)
        assert result.data is not None
        assert result.info is not None
        assert isinstance(result.info, ImageInfo)
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

    # Создаем реальные изображения
    image1 = test_dir / "image1.jpg"
    image2 = test_dir / "image2.png"
    PILImage.new('RGB', (100, 100), color='red').save(image1)
    PILImage.new('RGB', (100, 100), color='blue').save(image2)

    # Загружаем изображения
    images = image_loader.load_images(str(test_dir))

    # Проверяем результат
    assert len(images) == 2
    assert any("image1.jpg" in img[0] for img in images)
    assert any("image2.png" in img[0] for img in images)
    assert all(img[1] is not None for img in images)  # Изображения