# Файловая структура
```
face-clustering/
├── core/                          # Ядро приложения: абстракции и интерфейсы
│   ├── __init__.py
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── image_loader.py        # ImageLoader (абстрактный класс)
│   │   ├── face_detector.py       # FaceDetector (абстрактный класс)
│   │   ├── feature_extractor.py   # FeatureExtractor (абстрактный класс)
│   │   ├── cluster_analyzer.py    # ClusterAnalyzer (абстрактный класс)
│   │   ├── clusterer.py           # Clusterer (абстрактный класс)
│   │   ├── result_saver.py        # ResultSaver (абстрактный класс)
│   │   └── file_organizer.py      # FileOrganizer (абстрактный класс)
│   └── exceptions/
│       ├── __init__.py
│       ├── face_detection_error.py # FaceDetectionError
│       ├── clustering_error.py    # ClusteringError
│       └── file_handling_error.py # FileHandlingError
│
├── domain/                        # Модели предметной области
│   ├── __init__.py
│   ├── face.py                    # BoundingBox, Landmarks, Face
│   ├── image.py                   # Image, ImageInfo
│   ├── cluster.py                 # Cluster, ClusterResult
│   └── result.py                  # ClusteringResult
│
├── infrastructure/                # Конкретные реализации инфраструктуры
│   ├── __init__.py
│   ├── image/
│   │   ├── __init__.py
│   │   ├── os_image_loader.py     # OSImageLoader, ImagePreprocessor
│   │   └── image_preprocessor.py  # ImagePreprocessor (может быть вынесен отдельно)
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── insightface_detector.py # InsightFaceDetector
│   │   └── deepface_detector.py    # DeepFaceDetector
│   │
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── arcface_extractor.py    # ArcFaceExtractor
│   │   └── insightface_extractor.py # InsightFaceFeatureExtractor
│   │
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── silhouette_analyzer.py  # SilhouetteClusterAnalyzer
│   │   ├── elbow_analyzer.py       # ElbowClusterAnalyzer
│   │   └── kmeans_clusterer.py     # KMeansClusterer
│   │
│   └── persistence/
│       ├── __init__.py
│       ├── json_result_saver.py    # JSONResultSaver
│       └── file_system_organizer.py # FileSystemOrganizer
│
├── application/                   # Сервисы приложения и точки входа
│   ├── __init__.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── face_processing_service.py  # FaceProcessingService
│   │   ├── face_clustering_service.py  # FaceClusteringService
│   │   └── result_processing_service.py # ResultProcessingService
│   │
│   ├── use_cases/
│   │   ├── __init__.py
│   │   └── cluster_faces_use_case.py   # ClusterFacesUseCase
│   │
│   └── cli/
│       ├── __init__.py
│       ├── main.py                # Точка входа CLI приложения (не класс)
│       └── argument_parser.py     # ArgumentParser (класс)
│
├── utils/                         # Вспомогательные утилиты
│   ├── __init__.py
│   ├── math_utils.py              # MathUtils (статические методы)
│   ├── image_utils.py             # ImageUtils (статические методы)
│   ├── distance_calculator.py     # DistanceCalculator (статические методы)
│   └── visualization.py           # VisualizationUtils (статические методы)
│
├── config/                        # Конфигурация приложения
│   ├── __init__.py
│   ├── settings.py                # Settings (класс)
│   └── dependency_injector.py     # DependencyInjector (класс)
│
├── tests/                         # Тесты
│   ├── unit/
│   │   ├── core/
│   │   │   └── interfaces/        # Тесты для интерфейсов
│   │   ├── domain/
│   │   │   └── test_face.py       # Тесты для классов из face.py
│   │   └── infrastructure/
│   │       ├── detection/
│   │       │   └── test_insightface_detector.py # Тесты для InsightFaceDetector
│   │       └── extraction/
│   │           └── test_arcface_extractor.py    # Тесты для ArcFaceExtractor
│   ├── integration/
│   │   └── test_face_clustering_pipeline.py     # Тесты для интеграции компонентов
│   └── e2e/
│       └── test_end_to_end.py     # End-to-end тесты
│
└── requirements.txt               # Зависимости проекта
```