# переходим в директорию

# Разбиваем видео на кадры

ffmpeg -i *.mp4 frame_%05d.png
ffmpeg -i *.cmfv frame_%05d.png

# Вырезаем все лица из кадров
python /Users/user/PycharmProjects/FaceTools/cli/extract_faces.py -s . --padding=0.5
python /Users/user/PycharmProjects/FaceTools/cli/extract_faces.py -s . --padding=0.7

# Копирование уникальных
python /Users/user/PycharmProjects/FaceTools/cli/arcface_uniqueness.py  ./faces  ./unique_faces --threshold 0.85

# Группируем лица по схожести
python /Users/user/PycharmProjects/FaceTools/cli/make_matrix.py -s ./unique_faces  -o matrix

# С указанием порога
python /Users/user/PycharmProjects/FaceTools/cli/group_files.py -j matrix.json -o groups.json -d groups

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

# sam3
python /Users/user/PycharmProjects/FaceTools/cli/crop_face_square.py '/Users/user/Downloads/_person_Alexandra Agoston/unique_faces/347eef357ace9f772781895fde9307f8_face_1.jpg'
find . -type f -name "*.jpg" -exec python /Users/user/PycharmProjects/FaceTools/cli/crop_face_square.py {} --debug \;