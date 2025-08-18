import insightface
import numpy as np
from typing import List, Dict
from src.core.interfaces.face_detector import FaceDetector
from src.core.exceptions.face_detection_error import FaceDetectionError
from src.domain.face import Face, BoundingBox, Landmarks


class InsightFaceDetector(FaceDetector):
    """Реализация детектора лиц с использованием InsightFace."""

    def __init__(self, ctx_id: int = 0, det_size: Tuple[int, int] = (640, 640), det_thresh: float = 0.5):
        self.ctx_id = ctx_id
        self.det_size = det_size
        self.det_thresh = det_thresh
        self._initialize_model()

    def _initialize_model(self):
        """Инициализация модели детекции InsightFace."""
        try:
            self.model = insightface.app.FaceAnalysis(
                name='buffalo_l',
                root='./old_models',
                providers=['CUDAExecutionProvider' if self.ctx_id >= 0 else 'CPUExecutionProvider']
            )
            self.model.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
        except Exception as e:
            raise FaceDetectionError(f"Ошибка инициализации InsightFace модели: {str(e)}")

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        faces = self.model.get(image)
        return [self._convert_to_domain_face(face, image) for face in faces if face.det_score >= self.det_thresh]

    def _convert_to_domain_face(self, face_info, image) -> Face:
        """Конвертирует сырые данные из InsightFace в доменную модель."""
        h, w = image.shape[:2]

        # Конвертируем bbox
        x, y, x2, y2 = face_info.bbox.astype(int)
        bbox = BoundingBox(x, y, x2 - x, y2 - y)

        # Конвертируем landmarks
        landmarks = Landmarks(
            left_eye=(face_info.kps[0][0], face_info.kps[0][1]),
            right_eye=(face_info.kps[1][0], face_info.kps[1][1]),
            nose=(face_info.kps[2][0], face_info.kps[2][1]),
            mouth_left=(face_info.kps[3][0], face_info.kps[3][1]),
            mouth_right=(face_info.kps[4][0], face_info.kps[4][1])
        )

        return Face(
            bbox=bbox,
            landmarks=landmarks,
            embedding=face_info.embedding,
            confidence=face_info.det_score,
            orientation=self._determine_orientation(landmarks, w)
        )

    def _determine_orientation(self, landmarks: Landmarks, image_width: int) -> str:
        """Определяет ориентацию лица на основе расположения глаз."""
        eye_center_x = (landmarks.left_eye[0] + landmarks.right_eye[0]) / 2
        image_center_x = image_width / 2

        if abs(eye_center_x - image_center_x) / image_width > 0.2:
            return "profile" if eye_center_x < image_center_x else "profile_right"
        return "front"

    def get_model_info(self) -> Dict[str, str]:
        return {
            "name": "InsightFace",
            "model": "buffalo_l",
            "det_size": f"{self.det_size[0]}x{self.det_size[1]}",
            "threshold": str(self.det_thresh)
        }