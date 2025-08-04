# Установите необходимые библиотеки перед запуском
# pip install facenet-pytorch torch torchvision pillow numpy

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

def compare_faces(image_path1, image_path2):
    device = torch.device('cpu')


    # Инициализация MTCNN для обнаружения и выравнивания лиц
    mtcnn = MTCNN(
        image_size=512,
        margin=0,
        min_face_size=384,
        thresholds=[0.5, 0.5, 0.5],
        factor=0.5,
        post_process=True,
        device=device
    )

    # Инициализация предобученной модели FaceNet
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Загрузка и обработка первого изображения
    img1 = Image.open(image_path1).convert('RGB')
    face1 = mtcnn(img1)
    if face1 is None:
        raise ValueError(f"Лицо не обнаружено на изображении {image_path1}")

    # Загрузка и обработка второго изображения
    img2 = Image.open(image_path2).convert('RGB')
    face2 = mtcnn(img2)
    if face2 is None:
        raise ValueError(f"Лицо не обнаружено на изображении {image_path2}")

    # Добавление размерности батча и перемещение на устройство
    face1 = face1.unsqueeze(0).to(device)
    face2 = face2.unsqueeze(0).to(device)

    # Генерация эмбеддингов с помощью FaceNet
    with torch.no_grad():
        embedding1 = resnet(face1)
        embedding2 = resnet(face2)

    # Вычисление евклидова расстояния между эмбеддингами
    distance = (embedding1 - embedding2).norm().item()

    return distance

# Пример использования
if __name__ == "__main__":
    try:
        image1_path = "/Users/user/__!make_face/refer_bibi_1.png"  # Путь к первому изображению
        image2_path = "/Users/user/__!make_face/refer_bibi_2.png"  # Путь ко второму изображению
        #image2_path = "/Users/user/__!make_face/refer_bibi_1.png"

        #image1_path = "/Users/user/__!make_face/frame_Bibi/frame_02073_face_2.jpg"
        #image2_path = "/Users/user/__!make_face/frame_Bibi/frame_02085_face_2.jpg"

        distance = compare_faces(image1_path, image2_path)
        print(f"Евклидово расстояние между эмбеддингами: {distance:.4f}")

        if (distance < 0.7):
            print("✅ Лица принадлежат одному человеку")
        else:
            print("❌ Лица принадлежат разным людям")
    except Exception as e:
        print(f"Ошибка: {str(e)}")
