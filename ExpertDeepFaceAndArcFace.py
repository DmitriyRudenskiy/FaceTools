from deepface import DeepFace

# Сравнение двух изображений
result = DeepFace.verify(
    img1_path="/Users/user/__!make_face/refer_bibi_1.png",
    img2_path="/Users/user/__!make_face/refer_bibi_2.png",
    model_name="ArcFace",       # Современная модель (высокая точность)
    detector_backend="retinaface",  # Детектор лиц
    distance_metric="cosine"    # Метрика сравнения
)

print(f"Похожи: {result['verified']}")
print(f"Сходство: {1 - result['distance']:.4f}")
print(f"Порог: {result['threshold']}")

print(result)

