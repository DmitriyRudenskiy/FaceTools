import pytest
import os
from PIL import Image
from src.infrastructure.file.image_loader import ImageLoader


def test_image_loading():
    setup_test_directory = "./tests/images/group1"
    """Тестирует загрузку изображений из директории."""
    loader = ImageLoader()
    result = loader.load_images(str(setup_test_directory))

    # Вывод содержимого переменной result
    print("\nСодержимое result:", result)

    # Проверяем количество найденных изображений
    assert len(result) == 3

    # Проверяем наличие конкретных файлов
    filenames = [os.path.basename(path) for path in result]
    assert "image1.png" in filenames
    assert "image2.png" in filenames
    assert "image3.png" in filenames


def test_absolute_paths():
    """Тестирует, что возвращаются абсолютные пути."""
    setup_test_directory = "./tests/images/group1"
    loader = ImageLoader()
    result = loader.load_images(str(setup_test_directory))

    # Проверяем, что все пути абсолютные
    for path in result:
        assert os.path.isabs(path), f"Путь не является абсолютным: {path}"

    # Проверяем, что файлы существуют
    for path in result:
        assert os.path.exists(path), f"Файл не существует: {path}"
        assert os.path.isfile(path), f"Путь не является файлом: {path}"


def test_valid_image_files():
    """Тестирует, что все возвращенные файлы являются валидными изображениями."""
    setup_test_directory = "./tests/images/group1"
    loader = ImageLoader()
    result = loader.load_images(str(setup_test_directory))

    # Проверяем, что все файлы можно открыть как изображения
    for path in result:
        try:
            with Image.open(path) as img:
                # Проверяем, что изображение можно загрузить
                img.load()
            assert True, f"Файл не является валидным изображением: {path}"
        except Exception as e:
            pytest.fail(f"Файл {path} не является валидным изображением: {str(e)}")


def test_nonexistent_directory():
    """Тестирует обработку несуществующей директории."""
    loader = ImageLoader()
    # Должно вызывать исключение FileNotFoundError
    with pytest.raises(FileNotFoundError):
        loader.load_images("/nonexistent/directory")
