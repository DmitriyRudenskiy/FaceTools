# face_clustering_pipeline.py
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
import shutil
import warnings
warnings.filterwarnings('ignore')

# ... (остальные классы остаются без изменений: FaceDetector, FeatureExtractor, ClusterAnalyzer, FaceClusterer, ResultSaver) ...

class SelectedImagesCopier:
    """Отвечает за копирование выбранных изображений в отдельную директорию."""
    @staticmethod
    def copy_selected_images(
        faces_data: List[Dict[str, Any]], 
        cluster_labels: np.ndarray, 
        selected_labels: List[int], 
        output_directory: Path
    ) -> None:
        """
        Копирует изображения, соответствующие выбранным кластерам, в отдельную директорию.
        
        Args:
            faces_data: Список данных о лицах.
            cluster_labels: Метки кластеров для каждого лица.
            selected_labels: Список ID кластеров, изображения которых нужно скопировать.
            output_directory: Директория для сохранения выбранных изображений.
        """
        if not selected_labels:
            print("Список выбранных кластеров пуст. Копирование не выполнено.")
            return
            
        output_path = Path(output_directory)
        output_path.mkdir(exist_ok=True)
        
        copied_count = 0
        print(f"Копирование изображений из кластеров: {selected_labels}...")
        
        # Создаем множество для быстрого поиска
        selected_labels_set = set(selected_labels)
        
        # Словарь для отслеживания уже скопированных оригинальных файлов
        # Чтобы не копировать одно и то же изображение несколько раз, если на нем несколько лиц из выбранных кластеров
        copied_originals = set()
        
        for face_info, label in zip(faces_data, cluster_labels):
            if label in selected_labels_set:
                original_path_str = face_info['image_path']
                # Проверяем, копировали ли мы уже это изображение
                if original_path_str not in copied_originals:
                    original_path = Path(original_path_str)
                    new_filename = original_path.name
                    new_path = output_path / new_filename
                    
                    try:
                        shutil.copy2(original_path, new_path)
                        copied_originals.add(original_path_str)
                        copied_count += 1
                        # print(f"Скопировано: {original_path} -> {new_path}") # Опционально: логирование
                    except Exception as e:
                        print(f"Ошибка при копировании {original_path} в {new_path}: {e}")
        
        print(f"Успешно скопировано {copied_count} уникальных изображений в директорию: {output_directory}")

# ... (остальные классы остаются без изменений: FaceDetector, FeatureExtractor, ClusterAnalyzer, FaceClusterer, ResultSaver) ...

class FaceClusteringPipeline:
    """Основной класс для управления всем процессом кластеризации."""
    def __init__(self, ctx_id: int = 0, det_size: Tuple[int, int] = (640, 640)):
        self.detector = FaceDetector(det_size=det_size, ctx_id=ctx_id)
        self.feature_extractor = FeatureExtractor(self.detector)
        self.faces_data = []
        self.embeddings = []
        self.clusterer = None
        self.cluster_labels = None

    def run(
        self, 
        input_dir: Path, 
        output_dir: Path = None,
        output_json_name: str = "clusters.json",
        max_clusters: int = 20, 
        method: str = 'silhouette',
        save_images: bool = True,
        save_json: bool = True,
        copy_selected: bool = False, # Новый аргумент
        selected_clusters: List[int] = None, # Новый аргумент
        selected_output_dir: Path = None # Новый аргумент
    ) -> None:
        """
        Запускает полный процесс кластеризации.
        Args:
            input_dir: Директория с входными изображениями.
            output_dir: Директория для сохранения результатов. 
                        Если None, используется текущая директория для JSON и отдельная папка для изображений.
            output_json_name: Имя выходного JSON файла (используется, если output_dir не задан).
            max_clusters: Максимальное количество кластеров.
            method: Метод определения оптимального количества кластеров.
            save_images: Сохранять ли изображения в директории.
            save_json: Сохранять ли результаты в JSON файл.
            copy_selected: Копировать ли изображения из выбранных кластеров.
            selected_clusters: Список ID кластеров для копирования (если None, копируются все).
            selected_output_dir: Директория для сохранения выбранных изображений.
        """
        # 1. Извлечение признаков
        print("Обработка изображений...")
        self.faces_data, self.embeddings = self.feature_extractor.process_directory(input_dir)
        if len(self.embeddings) < 2:
            print("Найдено менее 2 лиц. Кластеризация невозможна.")
            return
        # 2. Анализ кластеров
        print("Поиск оптимального количества кластеров...")
        analyzer = ClusterAnalyzer(self.embeddings)
        optimal_k = analyzer.find_optimal_clusters(max_clusters=max_clusters, method=method)
        print(f"Оптимальное количество кластеров: {optimal_k}")
        # 3. Кластеризация
        self.clusterer = FaceClusterer(self.embeddings)
        self.cluster_labels = self.clusterer.cluster(optimal_k)
        # 4. Определяем пути для сохранения
        if output_dir is None:
            json_output_path = Path.cwd() / output_json_name
            images_output_path = Path.cwd() / "clustered_faces"
        else:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            json_output_path = output_path / "clusters.json"
            images_output_path = output_path
        # 5. Сохранение результатов
        if save_images:
            ResultSaver.save_clusters_to_directories(
                self.faces_data, self.cluster_labels, images_output_path
            )
        if save_json:
            ResultSaver.save_clusters_to_json(
                self.faces_data, self.cluster_labels, input_dir, json_output_path
            )
            
        # 6. Копирование выбранных изображений (новый шаг)
        if copy_selected and self.cluster_labels is not None:
             # Определяем директорию для выбранных изображений
            if selected_output_dir is None:
                if output_dir is None:
                     selected_output_path = Path.cwd() / "selected_images"
                else:
                     selected_output_path = Path(output_dir) / "selected_images"
            else:
                selected_output_path = Path(selected_output_dir)
            
            # Определяем, какие кластеры копировать
            if selected_clusters is None:
                # Если не указано, копируем все кластеры
                selected_labels = list(np.unique(self.cluster_labels))
                print(f"Копирование изображений из всех кластеров ({selected_labels})...")
            else:
                # Проверяем, существуют ли указанные кластеры
                available_clusters = set(np.unique(self.cluster_labels))
                selected_labels = [label for label in selected_clusters if label in available_clusters]
                missing_labels = [label for label in selected_clusters if label not in available_clusters]
                if missing_labels:
                    print(f"Предупреждение: следующие кластеры не найдены и будут пропущены: {missing_labels}")
                if not selected_labels:
                    print("Нет действительных кластеров для копирования.")
                else:
                    print(f"Копирование изображений из кластеров: {selected_labels}...")
            
            if selected_labels:
                SelectedImagesCopier.copy_selected_images(
                    self.faces_data, self.cluster_labels, selected_labels, selected_output_path
                )
            
        print("Кластеризация завершена!")

def main():
    """Точка входа в программу."""
    parser = argparse.ArgumentParser(description='Кластеризация лиц на изображениях')
    parser.add_argument('input_dir', help='Директория с изображениями')
    parser.add_argument('output_dir', nargs='?', default=None, help='Директория для сохранения результатов (по умолчанию: текущая директория)')
    parser.add_argument('--max_clusters', type=int, default=20, help='Максимальное количество кластеров')
    parser.add_argument('--method', choices=['silhouette', 'elbow'], default='silhouette', 
                       help='Метод определения оптимального количества кластеров')
    parser.add_argument('--ctx_id', type=int, default=0, help='ID устройства (-1 для CPU, 0 для GPU)')
    parser.add_argument('--no_images', action='store_true', help='Не сохранять изображения в директории')
    parser.add_argument('--no_json', action='store_true', help='Не сохранять JSON файл')
    parser.add_argument('--json_name', type=str, default='clusters.json', help='Имя выходного JSON файла (по умолчанию: clusters.json)')
    
    # Новые аргументы для копирования выбранных изображений
    parser.add_argument('--copy_selected', action='store_true', help='Копировать изображения из выбранных кластеров в отдельную директорию')
    parser.add_argument('--selected_clusters', type=int, nargs='*', help='Список ID кластеров для копирования (например, --selected_clusters 0 2 3). Если не указано, копируются все.')
    parser.add_argument('--selected_output_dir', type=str, help='Директория для сохранения выбранных изображений (по умолчанию: <output_dir>/selected_images или ./selected_images)')
    
    args = parser.parse_args()
    
    pipeline = FaceClusteringPipeline(ctx_id=args.ctx_id)
    pipeline.run(
        input_dir=Path(args.input_dir),
        output_dir=args.output_dir,
        output_json_name=args.json_name,
        max_clusters=args.max_clusters,
        method=args.method,
        save_images=not args.no_images,
        save_json=not args.no_json,
        copy_selected=args.copy_selected,          # Передаем новый аргумент
        selected_clusters=args.selected_clusters,  # Передаем новый аргумент
        selected_output_dir=args.selected_output_dir # Передаем новый аргумент
    )

if __name__ == "__main__":
    main()