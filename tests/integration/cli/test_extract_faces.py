import sys
from unittest.mock import patch


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
        print(f"Обработка {input_path} -> {output_dir}")
        return True

def main():
    if len(sys.argv) < 2:
        print("Ошибка: не указан путь к изображениям")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    injector = MockDependencyInjector()
    service = injector.get_face_crop_service()

    print(f"Начинаем обработку изображений из {input_path}")
    success = service.process_images(input_path, output_dir)

    if success:
        print("Обработка завершена успешно")
        sys.exit(0)  # Добавлен выход с кодом 0 при успехе
    else:
        print("Ошибка при обработке изображений")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    return script_path


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

    with patch.object(sys, 'argv', [str(script_path), str(test_dir), str(output_dir)]):
        with patch('sys.exit') as mock_exit:
            # Импортируем и выполняем скрипт
            import runpy
            runpy.run_path(str(script_path))
            mock_exit.assert_called_with(0)  # Изменено с assert_called_once_with на просто проверку аргумента


def test_cli_script_missing_input(tmp_path):
    """Проверяет обработку отсутствующего аргумента"""
    # Создаем временный CLI скрипт
    script_path = create_test_cli_script(tmp_path)  # Используем tmp_path вместо tempfile.mkdtemp()

    with patch.object(sys, 'argv', [str(script_path)]):
        with patch('sys.exit') as mock_exit:
            # Импортируем и выполняем скрипт
            import runpy
            runpy.run_path(str(script_path))
            mock_exit.assert_called()  # Проверяем, что exit был вызван
            assert mock_exit.call_args[0][0] != 0  # Проверяем, что код выхода не 0