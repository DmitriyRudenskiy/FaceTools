import datetime
import os
import sys
import zlib

# Константы для фильтров
ALLOWED_EXTENSIONS = {'.py', '.yaml'}
EXCLUDED_DIRS = {'__pycache__', '.git', 'venv', 'node_modules', '.vscode'}

def is_hidden(path):
    """Проверяет, является ли путь скрытым файлом или директорией."""
    if os.name == 'nt':  # Windows
        import ctypes
        try:
            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
            return bool(attrs & 2)  # FILE_ATTRIBUTE_HIDDEN
        except:
            return False
    else:  # Unix-like
        return os.path.basename(path).startswith('.')

def should_exclude(path):
    """Проверяет, следует ли исключить данный путь."""
    # Проверка на исключаемые директории
    for dirname in EXCLUDED_DIRS:
        if dirname in path.split(os.sep):
            return True
    # Проверка на скрытые файлы/директории
    if is_hidden(path):
        return True
    return False

def get_file_encoding(file_path):
    """Пытается определить кодировку файла."""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(2048)  # Читаем первые 2048 байт для определения кодировки

            # Попробуем UTF-8
            try:
                raw_data.decode('utf-8')
                return 'utf-8'
            except UnicodeDecodeError:
                pass

            # Попробуем Windows-1251 (для русскоязычных файлов)
            try:
                raw_data.decode('windows-1251')
                return 'windows-1251'
            except UnicodeDecodeError:
                pass

            # Попробуем UTF-16
            try:
                raw_data.decode('utf-16')
                return 'utf-16'
            except UnicodeDecodeError:
                pass

            return 'utf-8'  # по умолчанию
    except Exception as e:
        print(f"Ошибка при определении кодировки файла {file_path}: {e}")
        return 'utf-8'  # по умолчанию

def read_file_safe(file_path, max_size=1024*1024):
    """Читает файл с обработкой ошибок кодировки и проверкой размера."""
    try:
        # Проверяем размер файла
        file_size = os.path.getsize(file_path)
        if file_size > max_size:
            return None, f"Файл слишком большой ({file_size} байт)"

        # Пытаемся определить кодировку
        encoding = get_file_encoding(file_path)

        # Читаем файл с обработкой ошибок кодировки
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        return content, None
    except UnicodeDecodeError:
        # Если ошибка декодирования, пробуем читать как бинарный файл
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                # Проверяем, является ли файл бинарным (наличие нулевых байтов)
                if b'\x00' in content:
                    return None, "Бинарный файл"
                else:
                    # Если не бинарный, но есть ошибки декодирования, заменяем недопустимые символы
                    return content.decode(encoding, errors='replace'), None
        except Exception as e:
            return None, f"Ошибка чтения: {str(e)}"
    except Exception as e:
        return None, f"Ошибка чтения: {str(e)}"

def calculate_crc32(directory):
    """Вычисляет CRC32 хеш для директории (рекурсивно)."""
    crc = 0
    for root, dirs, files in os.walk(directory):
        # Сортируем директории и файлы для последовательного обхода
        dirs.sort()
        files.sort()

        # Исключаем ненужные директории
        dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d))]

        for file in files:
            file_path = os.path.join(root, file)
            if not should_exclude(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        data = f.read(65536)  # Читаем первые 64KB для хеширования
                        crc = zlib.crc32(data, crc)
                except Exception as e:
                    print(f"Ошибка при чтении файла для CRC: {file_path}, ошибка: {e}")
    return crc & 0xFFFFFFFF  # CRC32 возвращает знаковое число, поэтому обрезаем до 32 бит

def generate_project_context(project_dir, output_file='project_context.txt'):
    """Генерирует контекст проекта для передачи в LLM."""
    processed_files = 0
    skipped_dirs = set()
    warnings = []

    # Получаем мета-информацию
    generation_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    python_version = sys.version.split()[0]
    project_hash = calculate_crc32(project_dir)

    # Открываем выходной файл для записи
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # Записываем мета-информацию
        out_f.write("=== Мета-информация ===\n")
        out_f.write(f"Дата генерации: {generation_date}\n")
        out_f.write(f"Версия Python: {python_version}\n")
        out_f.write(f"Хеш-сумма проекта (CRC32): {project_hash:08X}\n")
        out_f.write("\n=== Содержимое файлов ===\n\n")

        # Рекурсивно обрабатываем директорию
        for root, dirs, files in os.walk(project_dir):
            # Сначала обрабатываем текущую директорию (файлы)
            for file in sorted(files):
                file_path = os.path.join(root, file)

                # Проверяем, нужно ли исключить этот файл
                if should_exclude(file_path):
                    continue

                # Проверяем расширение файла
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext not in ALLOWED_EXTENSIONS:
                    continue

                # Читаем файл
                content, error = read_file_safe(file_path)
                if error:
                    warnings.append(f"{file_path}: {error}")
                    continue

                # Записываем содержимое в выходной файл
                out_f.write(f"### {file_path} ###\n")
                out_f.write(content)
                out_f.write("\n===================\n\n")
                processed_files += 1

            # Затем обрабатываем поддиректории (если они не исключены)
            dirs[:] = sorted([d for d in dirs if not should_exclude(os.path.join(root, d))])
            for dirname in dirs:
                dir_path = os.path.join(root, dirname)
                if dirname in EXCLUDED_DIRS or should_exclude(dir_path):
                    skipped_dirs.add(dir_path)
                    dirs.remove(dirname)  # Удаляем из списка, чтобы не обрабатывать

        # Записываем статистику в конец файла
        out_f.write("\n=== Статистика ===\n")
        out_f.write(f"Обработано файлов: {processed_files}\n")
        out_f.write("\nПропущенные директории:\n")
        for dir_path in sorted(skipped_dirs):
            out_f.write(f"- {dir_path}\n")

        if warnings:
            out_f.write("\nПредупреждения:\n")
            for warning in warnings:
                out_f.write(f"- {warning}\n")

if __name__ == "__main__":
    # Определяем директорию проекта (на один уровень выше директории скрипта)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)  # Поднимаемся на один уровень выше

    print(f"Обработка директории проекта: {project_dir}")
    generate_project_context(project_dir)
    print("Контекст проекта успешно сгенерирован в файл project_context.txt")
