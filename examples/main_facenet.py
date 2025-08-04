import sys
import os

# Добавляем путь к src директории
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.expert_facenet import ExpertFaceNetMTCNN
from utils.compare_matrix import CompareMatrix

if __name__ == "__main__":
    comparator = ExpertFaceNetMTCNN()
    comparator.init([
        "/Users/user/__!make_face/refer_bibi_1.png",
        "/Users/user/__!make_face/refer_bibi_2.png",
        "/Users/user/__!make_face/frame_Bibi/frame_05719_face_1.jpg",
        "/Users/user/Downloads/ideogram_download_2025-08-03/0001_1_a-striking-artistic-portrait-of-a-woman-_55Pb9rG2QAePxTT6mPxVVA_niCnd0CrQhWLT9HfAwcBwA.jpeg",
        "/Users/user/Downloads/ideogram_download_2025-08-03/0003_4_a-captivating-beauty-portrait-of-a-strik_zy28OHyKTIWGe3DduVimeA_yVxFlr85TM-wGt_i9yt04w.jpeg"
    ])
    distance = comparator.compare(0, 1)
    print(len(comparator.storage))
    print(distance)

    matrix = CompareMatrix(len(comparator.storage))
    matrix.fill(comparator)
    matrix.display()