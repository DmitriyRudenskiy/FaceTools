# tests/unit/config/test_dependency_injector.py
import pytest
from src.config.dependency_injector import DependencyInjector
from src.application.services.face_crop_service import FaceCropService
from src.core.interfaces import FaceDetector, BoundingBoxProcessor, ImageLoader, FileOrganizer


def test_dependency_injector_creates_face_crop_service():
    """Проверяет, что инжектор создает экземпляр FaceCropService с необходимыми зависимостями"""
    injector = DependencyInjector()
    service = injector.get_face_crop_service()

    assert isinstance(service, FaceCropService)
    assert hasattr(service, 'file_organizer')
    assert hasattr(service, 'face_detector')
    assert hasattr(service, 'bbox_processor')
    assert hasattr(service, 'image_loader')
    assert isinstance(service.file_organizer, FileOrganizer)
    assert isinstance(service.face_detector, FaceDetector)
    assert isinstance(service.bbox_processor, BoundingBoxProcessor)
    assert isinstance(service.image_loader, ImageLoader)


def test_dependency_injector_creates_all_dependencies():
    """Проверяет, что инжектор создает все необходимые зависимости"""
    injector = DependencyInjector()

    # Проверяем создание всех компонентов
    file_organizer = injector.get_face_crop_service().file_organizer
    face_detector = injector.get_face_crop_service().face_detector
    bbox_processor = injector.get_face_crop_service().bbox_processor
    image_loader = injector.get_face_crop_service().image_loader

    assert isinstance(file_organizer, FileOrganizer)
    assert isinstance(face_detector, FaceDetector)
    assert isinstance(bbox_processor, BoundingBoxProcessor)
    assert isinstance(image_loader, ImageLoader)