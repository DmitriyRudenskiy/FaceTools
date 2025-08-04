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

class CompareMatrix:
    def __init__(self, size):
        self.matrix = np.zeros((size, size))

    def fill(self, comparator):
        """Заполнение матрицы: диагональ - NULL, заполняются только элементы i > j"""
        size = self.matrix.shape[0]
        for i in range(size):
            for j in range(size):
                if i == j:
                    self.matrix[i, j] = np.nan
                elif i > j:
                    self.matrix[i, j] = comparator.compare(i, j)
                    self.matrix[j, i] = self.matrix[i, j]

    def display(self):
        """Вывод матрицы на экран"""
        print("Текущая матрица:")
        print(self.matrix)


comparator = ExpertFaceNetMTCNN()
comparator.init(["/Users/user/__!make_face/refer_bibi_1.png", "/Users/user/__!make_face/refer_bibi_2.png",
                 "/Users/user/__!make_face/frame_Bibi/frame_05719_face_1.jpg",
                 "/Users/user/Downloads/ideogram_download_2025-08-03/0001_1_a-striking-artistic-portrait-of-a-woman-_55Pb9rG2QAePxTT6mPxVVA_niCnd0CrQhWLT9HfAwcBwA.jpeg",
                 "/Users/user/Downloads/ideogram_download_2025-08-03/0003_4_a-captivating-beauty-portrait-of-a-strik_zy28OHyKTIWGe3DduVimeA_yVxFlr85TM-wGt_i9yt04w.jpeg"])
distance = comparator.compare(0, 1)
print(len(comparator.storage))
print(distance)

matrix = CompareMatrix(len(comparator.storage))
matrix.fill(comparator)
matrix.display()
