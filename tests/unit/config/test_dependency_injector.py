# tests/unit/config/test_dependency_injector.py
def test_dependency_injector_creates_services():
    injector = DependencyInjector()
    service = injector.get_face_clustering_service()
    assert service is not None
    assert hasattr(service, 'process_images')