from typing import Dict, Any

class ImageData:
    """Класс для хранения данных об изображении"""

    def __init__(self, filename: str, full_path: str, sharpness: float, area: int,
                 width: int, height: int, has_face: bool):
        self.filename = filename
        self.full_path = full_path
        self.sharpness = sharpness
        self.area = area
        self.width = width
        self.height = height
        self.has_face = has_face

    def __repr__(self) -> str:
        return f"ImageData(filename='{self.filename}', sharpness={self.sharpness:.2f}, " \
               f"area={self.area}, width={self.width}, height={self.height}, has_face={self.has_face})"

    def to_dict(self) -> Dict[str, Any]:
        """Возвращает данные в виде словаря для JSON"""
        return {
            "filename": self.filename,
            "full_path": self.full_path,
            "sharpness": round(self.sharpness, 2),
            "area": self.area,
            "width": self.width,
            "height": self.height,
            "has_face": self.has_face
        }