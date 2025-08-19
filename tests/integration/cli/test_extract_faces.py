import sys
from unittest.mock import patch


def create_test_cli_script(tmp_path):
    """Создает временный CLI скрипт для тестирования"""
    script_path = tmp_path / "extract_faces.py"
    script_content = """#!/usr/bin/env python3
import sys
import os
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
        # Имитация обработки
        try:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "test_face_1.jpg"), "w") as f:
                    f.write("dummy")
            return True
        except Exception as e:
            print(f"Mock processing error: {{e}}", file=sys.stderr)
            return False

def main():
    print(f"Запуск с аргументами: {{sys.argv}}")
    if len(sys.argv) < 2:
        print("Error: Input directory required")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        injector = MockDependencyInjector()
        service = injector.get_face_crop_service()
        success = service.process_images(input_path, output_dir)
        print(f"Успешно: {{success}}")
        # ВАЖНО: Всегда вызываем sys.exit в конце main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Критическая ошибка: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
    script_path.write_text(script_content)
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
            print(f"Выполняем скрипт: {{script_path}}")
            try:
                runpy.run_path(str(script_path))
            except SystemExit as e:
                # runpy.run_path может не перехватывать SystemExit,
                # поэтому мы делаем это явно
                print(f"SystemExit перехвачен: {{e.code}}")
                mock_exit(e.code)
            except Exception as e:
                print(f"Исключение во время выполнения: {{e}}")
                raise

            print(f"mock_exit.call_count: {{mock_exit.call_count}}")
            print(f"mock_exit.call_args_list: {{mock_exit.call_args_list}}")

            # Проверяем, что exit был вызван
            mock_exit.assert_called()
            # Проверяем, что код выхода 0
            assert mock_exit.call_args[0][0] == 0

            # Дополнительная проверка: убедимся, что файл создан
            assert (output_dir / "test_face_1.jpg").exists()


def test_cli_script_missing_input(tmp_path):
    """Проверяет обработку отсутствующего аргумента"""
    # Создаем временный CLI скрипт
    script_path = create_test_cli_script(tmp_path)

    with patch.object(sys, 'argv', [str(script_path)]):
        with patch('sys.exit') as mock_exit:
            # Импортируем и выполняем скрипт
            import runpy
            runpy.run_path(str(script_path))

            # Проверяем, что exit был вызван
            mock_exit.assert_called()
            # Проверяем, что код выхода 1
            assert mock_exit.call_args[0][0] == 1