"""
Этот файл объединяет все интерфейсы в одном пространстве имен.
Позволяет импортировать интерфейсы напрямую из core.interfaces
"""

from .face_detector import BoundingBoxProcessor, FaceDetector
from .file_organizer import FileOrganizer
from .image_loader import ImageLoader
from .result_saver import ResultSaver

__all__ = [
    "FaceDetector",
    "BoundingBoxProcessor",
    "FileOrganizer",
    "ImageLoader",
    "ResultSaver",
]
