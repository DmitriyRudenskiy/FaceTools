# tests/integration/cli/test_extract_faces.py
import os
import sys
import pytest
from unittest.mock import patch, MagicMock


def create_test_cli_script(tmp_path):
    """Создает временный CLI скрипт для тестирования"""
    script_path = tmp_path / "extract_faces.py"
    script_content = """#!/usr/bin/env python3
import sys
from pathlib import Path

# Имитируем настройку пути
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# Имитируем импорт
class MockDependencyInjector:
    def get_face_crop_service(self):
        return MockFaceCropService()

class MockFaceCropService:
    def process_images(self, input_path, output_dir=None):
        # Создаем файлы для имитации обработки
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "test_face_1.jpg"), "w") as f:
                f.write("dummy")
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Input directory required")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    injector = MockDependencyInjector()
    service = injector.get_face_crop_service()
    success = service.process_images(input_path, output_dir)

    sys.exit(0 if success else 1)
"""
    script_path.write_text(script_content)
    return str(script_path)


def test_cli_script_success(tmp_path):
    """Проверяет успешное выполнение CLI скрипта"""
    # Создаем тестовую структуру
    test_dir = tmp_path / "test_images"
    test_dir.mkdir()
    (test_dir / "test.jpg").touch()

    # Создаем временный CLI скрипт
    script_path = create_test_cli_script(tmp_path)

    # Запускаем скрипт
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch.object(sys, 'argv', [script_path, str(test_dir), str(output_dir)]):
        with patch('sys.exit') as mock_exit:
            # Импортируем и выполняем скрипт
            import runpy
            runpy.run_path(script_path)
            mock_exit.assert_called_once_with(0)

    # Проверяем, что файлы созданы
    assert len(list(output_dir.glob("*.jpg"))) > 0


def test_cli_script_missing_input():
    """Проверяет обработку отсутствующего аргумента"""
    # Создаем временный CLI скрипт
    tmp_dir = tempfile.mkdtemp()
    script_path = create_test_cli_script(Path(tmp_dir))

    with patch.object(sys, 'argv', [script_path]):
        with patch('sys.exit') as mock_exit:
            # Импортируем и выполняем скрипт
            import runpy
            runpy.run_path(script_path)
            mock_exit.assert_called_once()
            # Проверяем, что код выхода не 0
            assert mock_exit.call_args[0][0] != 0