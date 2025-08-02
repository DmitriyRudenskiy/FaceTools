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

class FaceDetector:
    """Отвечает за обнаружение и извлечение признаков лиц из изображений."""
    
    def __init__(self, det_size: Tuple[int, int] = (640, 640), ctx_id: int = 0):
        self.det_size = det_size
        self.ctx_id = ctx_id
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)

    def extract_faces(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Извлекает лица и их эмбеддинги из одного изображения.
        
        Args:
            image_path: Путь к изображению.
            
        Returns:
            Список словарей с данными о лицах.
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Не удалось загрузить изображение: {image_path}")
                return []
            
            faces = self.app.get(image)
            faces_info = []
            for i, face in enumerate(faces):
                faces_info.append({
                    'embedding': face.normed_embedding,
                    'bbox': face.bbox,
                    'image_path': str(image_path),
                    'face_id': i
                })
            return faces_info
        except Exception as e:
            print(f"Ошибка при обработке изображения {image_path}: {e}")
            return []

class FeatureExtractor:
    """Управляет процессом извлечения признаков из директории с изображениями."""
    
    def __init__(self, detector: FaceDetector):
        self.detector = detector
        self.faces_data: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []

    def process_directory(self, directory_path: Path) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
        """
        Обрабатывает все изображения в директории.
        
        Args:
            directory_path: Путь к директории с изображениями.
            
        Returns:
            Кортеж из списка данных о лицах и списка эмбеддингов.
        """
        self.faces_data = []
        self.embeddings = []
        
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        for ext in supported_formats:
            image_paths.extend(directory_path.glob(f"*{ext}"))
            image_paths.extend(directory_path.glob(f"*{ext.upper()}"))
        
        print(f"Найдено {len(image_paths)} изображений для обработки")
        
        for i, image_path in enumerate(image_paths):
            print(f"Обработка изображения {i+1}/{len(image_paths)}: {image_path.name}")
            faces_info = self.detector.extract_faces(image_path)
            
            for face_info in faces_info:
                self.faces_data.append(face_info)
                self.embeddings.append(face_info['embedding'])
        
        print(f"Всего обнаружено лиц: {len(self.embeddings)}")
        return self.faces_data, self.embeddings

class ClusterAnalyzer:
    """Отвечает за анализ и определение оптимального количества кластеров."""
    
    def __init__(self, embeddings: List[np.ndarray]):
        if len(embeddings) == 0:
            raise ValueError("Нет данных для анализа.")
        self.embeddings_array = np.array(embeddings)
        self.optimal_clusters = None

    def find_optimal_clusters(self, max_clusters: int = 20, method: str = 'silhouette') -> int:
        """
        Находит оптимальное количество кластеров.
        
        Args:
            max_clusters: Максимальное количество кластеров для проверки.
            method: Метод поиска ('silhouette' или 'elbow').
            
        Returns:
            Оптимальное количество кластеров.
        """
        if len(self.embeddings_array) < 2:
            raise ValueError("Недостаточно лиц для кластеризации")
        
        if method == 'silhouette':
            return self._find_optimal_silhouette(max_clusters)
        elif method == 'elbow':
            return self._find_optimal_elbow(max_clusters)
        else:
            raise ValueError("Метод должен быть 'silhouette' или 'elbow'")

    def _find_optimal_silhouette(self, max_clusters: int) -> int:
        """Поиск оптимального количества кластеров методом силуэта."""
        silhouette_scores = []
        k_range = range(2, min(max_clusters + 1, len(self.embeddings_array)))
        
        print("Поиск оптимального количества кластеров методом силуэта...")
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.embeddings_array)
            silhouette_avg = silhouette_score(self.embeddings_array, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"k={k}, silhouette_score={silhouette_avg:.4f}")
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        self.optimal_clusters = optimal_k
        
        # Визуализация
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, silhouette_scores, 'bo-')
        plt.xlabel('Количество кластеров')
        plt.ylabel('Средний коэффициент силуэта')
        plt.title('Метод силуэта для определения оптимального количества кластеров')
        plt.grid(True)
        plt.show()
        
        return optimal_k

    def _find_optimal_elbow(self, max_clusters: int) -> int:
        """Поиск оптимального количества кластеров методом локтя."""
        inertias = []
        k_range = range(1, min(max_clusters + 1, len(self.embeddings_array)))
        
        print("Поиск оптимального количества кластеров методом локтя...")
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.embeddings_array)
            inertias.append(kmeans.inertia_)
            print(f"k={k}, inertia={kmeans.inertia_:.2f}")
        
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        elbow_point = np.argmax(diffs2) + 2
        optimal_k = max(2, min(elbow_point, len(k_range)))
        self.optimal_clusters = optimal_k
        
        # Визуализация
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, 'bo-')
        plt.xlabel('Количество кластеров')
        plt.ylabel('Инерция')
        plt.title('Метод локтя для определения оптимального количества кластеров')
        plt.grid(True)
        plt.show()
        
        return optimal_k

class FaceClusterer:
    """Выполняет кластеризацию лиц."""
    
    def __init__(self, embeddings: List[np.ndarray]):
        if len(embeddings) == 0:
            raise ValueError("Нет данных для кластеризации.")
        self.embeddings_array = np.array(embeddings)
        self.model = None

    def cluster(self, n_clusters: int) -> np.ndarray:
        """
        Выполняет кластеризацию.
        
        Args:
            n_clusters: Количество кластеров.
            
        Returns:
            Массив меток кластеров.
        """
        print(f"Выполнение кластеризации с {n_clusters} кластерами...")
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.model.fit_predict(self.embeddings_array)
        return cluster_labels

class ResultSaver:
    """Отвечает за сохранение результатов кластеризации."""
    
    @staticmethod
    def save_clusters_to_directories(
        faces_data: List[Dict[str, Any]], 
        cluster_labels: np.ndarray, 
        output_directory: Path
    ) -> None:
        """
        Сохраняет изображения лиц в отдельные директории по кластерам.
        
        Args:
            faces_data: Список данных о лицах.
            cluster_labels: Метки кластеров.
            output_directory: Директория для сохранения результатов.
        """
        output_path = Path(output_directory)
        output_path.mkdir(exist_ok=True)
        
        for cluster_id in range(len(np.unique(cluster_labels))):
            cluster_dir = output_path / f"cluster_{cluster_id}"
            cluster_dir.mkdir(exist_ok=True)
        
        for i, (face_info, label) in enumerate(zip(faces_data, cluster_labels)):
            original_path = Path(face_info['image_path'])
            cluster_dir = output_path / f"cluster_{label}"
            
            new_filename = f"face_{i}_{original_path.name}"
            new_path = cluster_dir / new_filename
            
            try:
                shutil.copy2(original_path, new_path)
            except Exception as e:
                print(f"Ошибка при копировании {original_path} в {new_path}: {e}")
        
        print(f"Результаты сохранены в директорию: {output_directory}")
        print("Статистика по кластерам:")
        unique, counts = np.unique(cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"  Кластер {cluster_id}: {count} лиц")

    @staticmethod
    def save_clusters_to_json(
        faces_data: List[Dict[str, Any]], 
        cluster_labels: np.ndarray, 
        base_directory: Path, 
        output_file: Path
    ) -> None:
        """
        Сохраняет информацию о кластерах в JSON файл.
        
        Args:
            faces_data: Список данных о лицах.
            cluster_labels: Метки кластеров.
            base_directory: Базовая директория с изображениями.
            output_file: Путь к выходному JSON файлу.
        """
        # Группируем данные по кластерам
        clusters_info = {}
        for face_info, label in zip(faces_data, cluster_labels):
            if label not in clusters_info:
                clusters_info[label] = {
                    'images': [],
                    'images_full_paths': []
                }
            
            original_path = Path(face_info['image_path'])
            filename = original_path.name
            full_path = str(base_directory / filename)
            
            clusters_info[label]['images'].append(filename)
            clusters_info[label]['images_full_paths'].append(full_path)
        
        # Формируем структуру для JSON
        output_data = {
            "timestamp": "2025-07-27T21:06:20",  # Можно заменить на datetime.now().isoformat()
            "total_groups": len(clusters_info),
            "unrecognized_count": 0,  # Можно реализовать логику для "нераспознанных"
            "groups": []
        }
        
        for label in sorted(clusters_info.keys()):
            group_data = clusters_info[label]
            # Определяем представителя группы (первое изображение)
            representative = group_data['images'][0] if group_data['images'] else ""
            representative_full_path = group_data['images_full_paths'][0] if group_data['images_full_paths'] else ""
            
            group_entry = {
                "id": int(label),
                "size": len(group_data['images']),
                "representative": representative,
                "representative_full_path": representative_full_path,
                "images": group_data['images'],
                "images_full_paths": group_data['images_full_paths']
            }
            output_data["groups"].append(group_entry)
        
        # Записываем в файл
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Файл с информацией о кластерах сохранен как: {output_file}")

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
        output_dir: Path = None, # Сделаем необязательным
        output_json_name: str = "clusters.json", # Имя файла JSON по умолчанию
        max_clusters: int = 20, 
        method: str = 'silhouette',
        save_images: bool = True,
        save_json: bool = True
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
        # Если output_dir не задан, определяем пути по умолчанию
        if output_dir is None:
            # Для JSON файла используем текущую директорию
            json_output_path = Path.cwd() / output_json_name
            # Для изображений создаем папку в текущей директории
            images_output_path = Path.cwd() / "clustered_faces"
        else:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True) # Создаем директорию, если не существует
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
    
    args = parser.parse_args()
    
    pipeline = FaceClusteringPipeline(ctx_id=args.ctx_id)
    pipeline.run(
        input_dir=Path(args.input_dir),
        output_dir=args.output_dir, # Может быть None
        output_json_name=args.json_name,
        max_clusters=args.max_clusters,
        method=args.method,
        save_images=not args.no_images,
        save_json=not args.no_json
    )

if __name__ == "__main__":
    main()