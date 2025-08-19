from dataclasses import dataclass
from typing import Tuple


@dataclass
class ImageInfo:
    """Метаданные изображения"""

    path: str
    size: Tuple[int, int]  # (width, height)
    format: str


@dataclass
class Image:
    """Доменная модель изображения"""

    data: any  # PIL.Image или numpy array
    info: ImageInfo
