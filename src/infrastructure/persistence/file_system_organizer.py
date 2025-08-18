from src.core.interfaces.file_organizer import FileOrganizer
import os
import shutil
from typing import List, Dict


class FileSystemOrganizer(FileOrganizer):
    """Работа с файловой системой"""

    def get_image_files(self, path: str) -> List[str]:
        """Возвращает список путей к изображениям в указанном пути"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}

        if not os.path.exists(path):
            return []

        if os.path.isfile(path):
            _, ext = os.path.splitext(path.lower())
            return [path] if ext in image_extensions else []

        return [
            os.path.join(root, file)
            for root, _, files in os.walk(path)
            for file in files
            if os.path.splitext(file)[1].lower() in image_extensions
        ]

    def organize_by_clusters(self, clusters: List[Dict], destination: str) -> None:
        """Организует файлы по кластерам в отдельные директории"""
        os.makedirs(destination, exist_ok=True)
        print(f"Создана директория для групп: {destination}")

        for cluster in clusters:
            # Создаем имя директории на основе представителя
            representative_name = os.path.splitext(cluster["representative"])[0]
            safe_name = "".join(c for c in representative_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            if not safe_name:
                safe_name = f"Group_{cluster['id']}"

            group_dir = os.path.join(destination, safe_name)
            os.makedirs(group_dir, exist_ok=True)

            # Копируем файлы в директорию кластера
            copied = 0
            for full_path in cluster["members_paths"]:
                try:
                    filename = os.path.basename(full_path)
                    dest_path = os.path.join(group_dir, filename)
                    shutil.copy2(full_path, dest_path)
                    copied += 1
                except Exception as e:
                    print(f"Ошибка копирования {filename}: {str(e)}")

            print(f"Группа '{safe_name}': скопировано {copied} файлов")

    def create_directory(self, path: str) -> None:
        """Создает директорию, если она не существует"""
        os.makedirs(path, exist_ok=True)

    def exists(self, path: str) -> bool:
        """Проверяет существование пути"""
        return os.path.exists(path)

    def is_directory(self, path: str) -> bool:
        """Проверяет, является ли путь директорией"""
        return os.path.isdir(path)

    def get_directory(self, path: str) -> str:
        """Возвращает директорию, содержащую файл"""
        return os.path.dirname(path)

    def get_basename(self, path: str) -> str:
        """Возвращает базовое имя файла без расширения"""
        return os.path.splitext(os.path.basename(path))[0]

    def save(self, image: any, path: str) -> None:
        """Сохраняет изображение в файл"""
        dir_path = os.path.dirname(path)
        if dir_path and not self.exists(dir_path):
            self.create_directory(dir_path)

        image.save(path)