from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BoundingBox:
    """Границы обнаруженного лица"""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_list(self) -> List[float]:
        """Преобразует bounding box в список координат"""
        return [self.x1, self.y1, self.x2, self.y2]

    def area(self) -> float:
        """Вычисляет площадь bounding box"""
        return self.width * self.height

    def contains_point(self, x: float, y: float) -> bool:
        """Проверяет, содержится ли точка в bounding box"""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class Landmarks:
    """Ключевые точки лица"""

    left_eye: Tuple[float, float]
    right_eye: Tuple[float, float]
    nose: Tuple[float, float]
    mouth_left: Tuple[float, float]
    mouth_right: Tuple[float, float]

    def get_eye_distance(self) -> float:
        """Вычисляет расстояние между глазами"""
        from math import sqrt

        x1, y1 = self.left_eye
        x2, y2 = self.right_eye
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def get_all_points(self) -> List[Tuple[float, float]]:
        """Возвращает все точки в виде списка"""
        return [
            self.left_eye,
            self.right_eye,
            self.nose,
            self.mouth_left,
            self.mouth_right,
        ]


@dataclass
class Image:
    """Доменная модель изображения"""

    path: str
    width: int
    height: int
    format: Optional[str] = None
    faces: List["Face"] = field(default_factory=list)

    @property
    def aspect_ratio(self) -> float:
        """Вычисляет соотношение сторон изображения"""
        return self.width / self.height if self.height > 0 else 0

    def add_face(self, face: "Face") -> None:
        """Добавляет лицо в список обнаруженных на изображении"""
        self.faces.append(face)

    def get_largest_face(self) -> Optional["Face"]:
        """Возвращает самое большое лицо на изображении"""
        if not self.faces:
            return None
        return max(self.faces, key=lambda f: f.bounding_box.area())


@dataclass
class Face:
    """Доменная модель лица"""

    bounding_box: BoundingBox
    landmarks: Landmarks
    embedding: List[float]
    image: Image
    is_reference: bool = False
    confidence: float = 1.0

    def __post_init__(self):
        """Проверяет корректность данных после инициализации"""
        if not (0 <= self.confidence <= 1.0):
            self.confidence = 1.0

    @property
    def face_id(self) -> str:
        """Генерирует уникальный идентификатор лица"""
        return (
            f"{self.image.path}#{self.bounding_box.x1:.0f},{self.bounding_box.y1:.0f}"
        )

    def distance_to(self, other: "Face") -> float:
        """Вычисляет расстояние между эмбеддингами двух лиц"""
        if len(self.embedding) != len(other.embedding):
            raise ValueError("Эмбеддинги имеют разную длину")

        # Евклидово расстояние
        return sum((a - b) ** 2 for a, b in zip(self.embedding, other.embedding)) ** 0.5

    def is_similar_to(self, other: "Face", threshold: float = 0.6) -> bool:
        """Проверяет, похожи ли два лица"""
        return self.distance_to(other) < threshold

    def get_eye_center(self) -> Tuple[float, float]:
        """Вычисляет центр между глазами"""
        left_eye = self.landmarks.left_eye
        right_eye = self.landmarks.right_eye
        return ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    def get_face_size_ratio(self) -> float:
        """Возвращает отношение размера лица к размеру изображения"""
        face_area = self.bounding_box.area()
        image_area = self.image.width * self.image.height
        return face_area / image_area if image_area > 0 else 0
