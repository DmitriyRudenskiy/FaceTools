"""Module for analyzing image sharpness using OpenCV."""

from typing import Tuple
import cv2

class ImageSharpnessAnalyzer:
    """Класс для анализа резкости лица на изображениях."""

    # Загружаем каскад для обнаружения лиц один раз при инициализации класса
    # Using ignore comment for mypy as cv2.data is not properly typed
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # type: ignore[attr-defined]
    )

    @staticmethod
    def calculate_face_sharpness(image_path: str) -> float:
        """
        Вычисляет резкость лица на изображении методом Лапласиана.

        Args:
            image_path (str): Путь к изображению.

        Returns:
            float: Значение резкости лица (дисперсия Лапласиана).
        """
        image = cv2.imread(image_path)
        if image is None:
            return 0.0

        # Проверка, загружен ли каскад
        if ImageSharpnessAnalyzer.face_cascade.empty():
            print("Ошибка: Не удалось загрузить каскад для обнаружения лиц")
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Обнаружение лиц на изображении
        faces = ImageSharpnessAnalyzer.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Если лица не найдены, возвращаем 0.0
        if len(faces) == 0:
            return 0.0

        # Выбираем наибольшее лицо (по площади)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        # Вырезаем область лица
        face_roi = gray[y:y + h, x:x + w]

        # Вычисляем резкость только для области лица
        laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
        sharpness: float = laplacian.var()
        return sharpness

    @staticmethod
    def get_image_info(image_path: str) -> Tuple[int, int, int, bool]:
        """
        Получает информацию об изображении и наличии лица.

        Args:
            image_path (str): Путь к изображению.

        Returns:
            Tuple[int, int, int, bool]: Ширина, высота, площадь, наличие лица.
        """
        image = cv2.imread(image_path)
        if image is None:
            return 0, 0, 0, False

        height, width = image.shape[:2]
        area = width * height

        # Проверяем наличие лица
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = ImageSharpnessAnalyzer.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        has_face = len(faces) > 0
        return width, height, area, has_face
