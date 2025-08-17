from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class Landmarks:
    left_eye: Tuple[float, float]
    right_eye: Tuple[float, float]
    nose: Tuple[float, float]
    mouth_left: Tuple[float, float]
    mouth_right: Tuple[float, float]


@dataclass
class Face:
    bbox: BoundingBox
    landmarks: Optional[Landmarks] = None
    embedding: Optional[np.ndarray] = None
    confidence: float = 0.0
    orientation: str = "front"
    image_path: str = ""