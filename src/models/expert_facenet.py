import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

class ExpertFaceNetMTCNN:
    def __init__(self, device=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.mtcnn = MTCNN(
            image_size=512,
            margin=0,
            min_face_size=384,
            thresholds=[0.5, 0.5, 0.5],
            factor=0.5,
            post_process=True,
            device=self.device
        )

        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.storage = []

    def init(self, image_paths):
        for image_path in image_paths:
            try:
                img = Image.open(image_path).convert('RGB')
                face = self.mtcnn(img)
                if face is None:
                    raise ValueError(f"Лицо не обнаружено на изображении {image_path}")

                face = face.unsqueeze(0).to(self.device)
                embedding = self.resnet(face)

                self.storage.append((embedding, image_path))
            except Exception as e:
                print(e)

    def compare(self, index1, index2):
        emb1 = self.storage[index1][0]
        emb2 = self.storage[index2][0]
        distance = (emb1 - emb2).norm().item()
        return distance