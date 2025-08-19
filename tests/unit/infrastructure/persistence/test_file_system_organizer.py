import pytest
from unittest.mock import MagicMock, patch
from src.infrastructure.persistence.file_system_organizer import FileSystemOrganizer


@pytest.fixture
def file_organizer():
    return FileSystemOrganizer()


def test_exists(file_organizer):
    """Проверяет проверку существования файла/директории"""
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        assert file_organizer.exists("/test/path") is True

        mock_exists.return_value = False
        assert file_organizer.exists("/test/path") is False


def test_save(file_organizer, tmp_path):
    """Проверяет сохранение файла"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_image = MagicMock()
    file_organizer.save(mock_image, str(output_dir / "test.jpg"))

    # Проверяем, что makedirs был вызван
    assert os.path.exists(output_dir)
    # Проверяем, что save был вызван
    mock_image.save.assert_called_once_with(str(output_dir / "test.jpg"))


def test_get_basename(file_organizer):
    """Проверяет получение базового имени файла"""
    assert file_organizer.get_basename("/path/to/test.jpg") == "test"
    assert file_organizer.get_basename("test.png") == "test"
    assert file_organizer.get_basename("/path/to/test") == "test"