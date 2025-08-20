"""
Unit tests for the collage_creator module.

This module contains tests for the CollageCreator class, ensuring that
collage creation and grid handling work correctly under various conditions.
"""

import os
from unittest.mock import MagicMock, patch, call
from typing import List
from PIL import Image  # type: ignore[import-untyped]
from src.utils.collage_creator import CollageCreator  # pylint: disable=import-error

TEST_OUTPUT_DIR = "test_output"


def get_test_images(count: int = 16) -> List[Image.Image]:
    """Generate test images with different colors."""
    return [
        Image.new('RGB', (100, 100), color=color)
        for color in ['red', 'green', 'blue', 'yellow', 'purple', 'orange',
                      'pink', 'brown', 'gray', 'cyan', 'magenta', 'white',
                      'black', 'navy', 'teal', 'olive'][:count]
    ]


def test_collage_grid_2_no_remainder() -> None:
    """Test 2x2 grid collage creation without remainder."""
    creator = CollageCreator()
    images = get_test_images(4)

    with patch("src.utils.collage_creator.Image.new") as mock_new:
        mock_new.return_value = MagicMock()

        for img in images:
            img.resize = MagicMock(return_value=img)

        grid_size = 2
        creator.create_collage(images, grid_size)

        cell_size = 1024 // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                img = images[i * grid_size + j]
                img.resize.assert_any_call(
                    (cell_size, cell_size),
                    Image.Resampling.LANCZOS
                )

        mock_new.return_value.paste.assert_has_calls([
            call(images[0], (0, 0)),
            call(images[1], (cell_size, 0)),
            call(images[2], (0, cell_size)),
            call(images[3], (cell_size, cell_size))
        ], any_order=False)


def test_collage_grid_3_with_remainder() -> None:
    """Test 3x3 grid collage creation with remainder handling."""
    creator = CollageCreator()
    images = get_test_images(9)

    with patch("src.utils.collage_creator.Image.new") as mock_new:
        mock_new.return_value = MagicMock()

        for img in images:
            img.resize = MagicMock(return_value=img)

        grid_size = 3
        creator.create_collage(images, grid_size)

        cell_size = 1024 // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                # Используем прямое обращение к элементу списка вместо промежуточной переменной
                images[i * grid_size + j].resize.assert_any_call(
                    (
                        cell_size + (1 if j == grid_size - 1 else 0),
                        cell_size + (1 if i == grid_size - 1 else 0)
                    ),
                    Image.Resampling.LANCZOS
                )

        mock_new.return_value.paste.assert_has_calls([
            call(images[i * grid_size + j], (j * cell_size, i * cell_size))
            for i in range(grid_size)
            for j in range(grid_size)
        ], any_order=False)


def test_create_collages_multiple() -> None:
    """Test creating multiple collages with sufficient images for some grids."""
    creator = CollageCreator()
    images = get_test_images(10)

    with patch("src.utils.collage_creator.os.makedirs"), \
            patch("src.utils.collage_creator.os.path.exists", return_value=False), \
            patch("src.utils.collage_creator.random.sample", side_effect=lambda seq, n: seq[:n]), \
            patch("src.utils.collage_creator.Image.new", return_value=MagicMock()), \
            patch("PIL.Image.Image.save"):

        collages = creator.create_collages(images, "/fake/output")
        assert len(collages) == 2
        basenames = {os.path.basename(p) for p in collages}
        assert "collage_2x2.jpg" in basenames
        assert "collage_3x3.jpg" in basenames
        assert "collage_4x4.jpg" not in basenames
        assert creator.errors == ["Пропущена сетка 4x4 (недостаточно изображений)"]


def test_create_collages_skips_insufficient_images() -> None:
    """Test skipping collage creation when images are insufficient for all grids."""
    creator = CollageCreator()
    images = get_test_images(3)

    with patch("src.utils.collage_creator.os.makedirs"), \
            patch("src.utils.collage_creator.os.path.exists", return_value=False), \
            patch("src.utils.collage_creator.random.sample"), \
            patch("src.utils.collage_creator.Image.new"), \
            patch("PIL.Image.Image.save"):

        assert not creator.create_collages(images, "/fake/output")
        assert creator.errors == [
            "Пропущена сетка 2x2 (недостаточно изображений)",
            "Пропущена сетка 3x3 (недостаточно изображений)",
            "Пропущена сетка 4x4 (недостаточно изображений)"
        ]


def test_create_collage_handles_empty_images() -> None:
    """Test collage creation with an empty image list."""
    creator = CollageCreator()

    with patch("src.utils.collage_creator.Image.new") as mock_new:
        collage = MagicMock()
        mock_new.return_value = collage

        creator.create_collage([], grid_size=2)
        collage.paste.assert_not_called()
        mock_new.assert_called_with('RGB', (1024, 1024))


def test_create_collages_success() -> None:
    """Test successful creation of collages with sufficient images."""
    creator = CollageCreator()
    images = get_test_images(16)

    with patch("os.makedirs") as mock_makedirs, \
            patch("os.path.exists", return_value=False), \
            patch("PIL.Image.Image.save") as mock_save:
        with patch('random.sample', side_effect=lambda x, k: x[:k]):
            creator.create_collages(images, TEST_OUTPUT_DIR)

        mock_makedirs.assert_called_once_with(TEST_OUTPUT_DIR)
        assert mock_save.call_count == 3
        assert {call_args[0][0] for call_args in mock_save.call_args_list} == {
            f"{TEST_OUTPUT_DIR}/collage_2x2.jpg",
            f"{TEST_OUTPUT_DIR}/collage_3x3.jpg",
            f"{TEST_OUTPUT_DIR}/collage_4x4.jpg"
        }
        assert not creator.errors


def test_create_collages_existing_dir() -> None:
    """Test collage creation when output directory exists."""
    creator = CollageCreator()
    images = get_test_images(16)

    with patch("os.makedirs") as mock_makedirs, \
            patch("os.path.exists", return_value=True), \
            patch("PIL.Image.Image.save") as mock_save:
        with patch('random.sample', side_effect=lambda x, k: x[:k]):
            creator.create_collages(images, TEST_OUTPUT_DIR)

        mock_makedirs.assert_not_called()
        assert mock_save.call_count == 3


def test_create_collages_insufficient_images_records_errors() -> None:
    """Test error messages are properly recorded in errors list."""
    creator = CollageCreator()
    few_images = get_test_images(3)

    with patch("os.makedirs"), \
            patch("os.path.exists", return_value=False):
        creator.create_collages(few_images, TEST_OUTPUT_DIR)

    assert creator.errors == [
        "Пропущена сетка 2x2 (недостаточно изображений)",
        "Пропущена сетка 3x3 (недостаточно изображений)",
        "Пропущена сетка 4x4 (недостаточно изображений)"
    ]
