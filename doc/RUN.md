# переходим в директорию

# Разбиваем видео на кадры
ffmpeg -i 'input.mp4'  -q:v 2 frame_%05d.jpg

# Вырезаем все лица из кадров
python /Users/user/PycharmProjects/FaceTools/cli/extract_faces.py -s .

# Группируем лица по схожести
python /Users/user/PycharmProjects/FaceTools/cli/deepface_face_cluster.py -s './faces/1' -d ./grouped_photos

# С указанием порога
python /Users/user/PycharmProjects/FaceTools/cli/group_files.py deepface_groups.json groups --threshold 0.3

# Открываем страницу и загружаем созданный файл
groups.json -> public/show_groups.html

# Анализируем группу
python /Users/user/PycharmProjects/FaceTools/cli/deepface_face_cluster.py -s '/Users/user/Downloads/_face_Lea E/grouped_photos/lea_e_00271_face_1'

/Applications/Blender.app/Contents/MacOS/Blender \
--background \
--python "/Users/user/PycharmProjects/FaceTools/cli/render_script.py" \
-- "/Users/user/Downloads/20250913_001.glb" \
--verbose 5

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

