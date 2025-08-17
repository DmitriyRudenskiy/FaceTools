from dataclasses import dataclass
from typing import List, Tuple


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


@dataclass
class Landmarks:
    """Ключевые точки лица"""
    left_eye: Tuple[float, float]
    right_eye: Tuple[float, float]
    nose: Tuple[float, float]
    mouth_left: Tuple[float, float]
    mouth_right: Tuple[float, float]


@dataclass
class Face:
    """Доменная модель лица"""
    bounding_box: BoundingBox
    landmarks: Landmarks
    embedding: List[float]
    image: 'Image'