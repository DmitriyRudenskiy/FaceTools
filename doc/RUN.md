# Разбиваем видео на кадры

# Вырезаем все лица из кадров
python /Users/user/PycharmProjects/FaceCluster/application/cli/extract_faces.py './frames'

# Группируем лица по схожести
python /Users/user/PycharmProjects/FaceCluster/application/cli/face_cluster.py '/Users/user/Downloads/_face_Lea E/faces'