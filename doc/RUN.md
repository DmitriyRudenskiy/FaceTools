# Разбиваем видео на кадры

# Вырезаем все лица из кадров
python /Users/user/PycharmProjects/FaceCluster/cli/extract_faces.py '/Users/user/Downloads/_face_Lea E'

# Группируем лица по схожести
python /Users/user/PycharmProjects/FaceTools/cli/face_cluster.py -s '/Users/user/Downloads/_face_Lea E/faces' -d ./grouped_photos

# Открываем страницу и загружаем созданный файл
groups.json -> public/show_groups.html

# Анализируем группу
python /Users/user/PycharmProjects/FaceTools/cli/deepface_face_cluster.py -s '/Users/user/Downloads/_face_Lea E/grouped_photos/lea_e_00271_face_1'



python /Users/user/PycharmProjects/FaceTools/cli/collage.py -s /Users/user/Downloads/1

### Особенности реализации:
1. Классы разделены по своим ответственностям:
   - `ImageLoader` отвечает за загрузку изображений
   - `CollageCreator` отвечает за создание коллажей

2. Функциональность:
   - Загрузка изображений из указанной директории
   - Создание коллажей размером 2x2, 3x3 и 4x4
   - Случайное перемешивание изображений перед созданием коллажей
   - Сохранение результатов в указанную директорию