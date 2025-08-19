import os  # Добавлен импорт os
from unittest.mock import patch

import pytest
from PIL import Image as PILImage

from src.infrastructure.persistence.file_system_organizer import \
    FileSystemOrganizer


@pytest.fixture
def file_organizer():
    return FileSystemOrganizer()


def test_exists(file_organizer):
    """Проверяет проверку существования файла/директории"""
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        assert file_organizer.exists("/test/path")
        mock_exists.return_value = False
        assert not file_organizer.exists("/test/path")


def test_save(file_organizer, tmp_path):
    """Проверяет сохранение файла"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Создаем реальное изображение
    mock_image = PILImage.new('RGB', (100, 100), color='red')

    # Сохраняем изображение
    file_organizer.save(mock_image, str(output_dir / "test.jpg"))

    # Проверяем, что файл существует
    assert os.path.exists(output_dir / "test.jpg")


def test_get_basename(file_organizer):
    """Проверяет получение базового имени файла"""
    assert file_organizer.get_basename("/test/path/image.jpg") == "image"
    assert file_organizer.get_basename("image.png") == "image"