from typing import Any


class ImageInfo:
    def __init__(self, width: int, height: int, format: str):
        self.width = width
        self.height = height
        self.format = format


class Image:
    def __init__(self, data: Any, info: ImageInfo):
        self.data = data
        self.info = info
