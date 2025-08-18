import os
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage

from src.core.exceptions.file_handling_error import FileHandlingError
from src.core.interfaces.image_loader import ImageLoader
from src.domain.image_model import Image, ImageInfo


class OSImageLoader(ImageLoader):
    """Реализация загрузки изображений через операционную систему."""

    SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor

    def validate_directory(self, directory: str) -> bool:
        return os.path.isdir(directory) and os.access(directory, os.R_OK)

    def load_images(self, directory: str) -> List[Tuple[str, np.ndarray]]:
        if not self.validate_directory(directory):
            raise FileHandlingError(
                f"Директория {directory} не существует или недоступна"
            )

        images = []
        for filename in os.listdir(directory):
            if filename.lower().endswith(self.SUPPORTED_EXTENSIONS):
                file_path = os.path.join(directory, filename)
                try:
                    image = cv2.imread(file_path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # Применяем предварительную обработку, если она задана
                        if self.preprocessor:
                            image = self.preprocessor.preprocess(image)

                        images.append((file_path, image))
                except Exception as e:
                    raise FileHandlingError(
                        f"Ошибка загрузки файла {file_path}: {str(e)}"
                    )

        return images

    """Загрузка изображений через PIL"""

    def load(self, path: str) -> Image:
        pil_image = PILImage.open(path)

        return Image(
            data=pil_image,
            info=ImageInfo(path=path, size=pil_image.size, format=pil_image.format),
        )
