import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Добавляем корневую директорию проекта в sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"[DEBUG] Добавлена корневая директория в sys.path: {project_root}")

# Фикстура для мокирования зависимостей FaceCropService
@pytest.fixture
def mock_dependencies():
    """Создает моки для всех зависимостей FaceCropService"""
    return {
        'file_organizer': MagicMock(),
        'face_detector': MagicMock(),
        'bbox_processor': MagicMock(),
        'image_loader': MagicMock()
    }