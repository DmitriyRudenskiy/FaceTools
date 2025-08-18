"""
Модуль для организации файлов по группам в отдельные каталоги.
"""

import os
import shutil
import time


class GroupOrganizer:
    """Класс для организации файлов по группам в отдельные каталоги."""

    def __init__(self, groups_data, destination_directory):
        """
        Инициализирует GroupOrganizer.
        Args:
            groups_data (list): Список словарей с данными о группах, полученный от ImageGrouper.
            destination_directory (str): Путь к каталогу, где будут созданы подкаталоги для групп.
        """
        self.groups_data = groups_data
        self.destination_directory = destination_directory
        # Создаем основную директорию, если она не существует
        os.makedirs(self.destination_directory, exist_ok=True)

    def organize(self):
        """
        Создает каталоги для каждой группы и копирует в них файлы.
        Каталог называется по имени представителя группы.
        """
        print("=== Начало организации файлов по группам ===")
        start_time = time.time()
        total_copied = 0
        for group_data in self.groups_data:
            group_id = group_data["id"]
            # Используем имя файла представителя (без расширения) как имя каталога
            representative_name = os.path.splitext(group_data["representative"])[0]
            # Очищаем имя от недопустимых символов для имен файлов/каталогов
            safe_group_name = "".join(
                c for c in representative_name if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()
            # Если имя оказалось пустым, используем ID группы
            if not safe_group_name:
                safe_group_name = f"Group_{group_id}"
            group_directory_path = os.path.join(
                self.destination_directory, safe_group_name
            )
            print(f"Создаю каталог для группы {group_id}: {group_directory_path}")
            # Создаем каталог для группы
            try:
                os.makedirs(group_directory_path, exist_ok=True)
            except OSError as e:
                print(f"Ошибка создания каталога {group_directory_path}: {e}")
                continue  # Пропускаем эту группу, если не удалось создать каталог

            # Копируем файлы группы в созданный каталог
            copied_count = 0
            for full_path in group_data["images_full_paths"]:
                try:
                    filename = os.path.basename(full_path)
                    destination_file_path = os.path.join(group_directory_path, filename)
                    # Копируем файл
                    shutil.copy2(full_path, destination_file_path)
                    copied_count += 1
                    total_copied += 1
                except Exception as e:
                    print(
                        f"Ошибка копирования файла {full_path} в {group_directory_path}: {e}"
                    )
            print(f"  Скопировано файлов в группу '{safe_group_name}': {copied_count}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("=== Организация файлов завершена ===")
        print(f"Всего скопировано файлов: {total_copied}")
        print(f"Время на организацию: {elapsed_time:.2f} секунд")
