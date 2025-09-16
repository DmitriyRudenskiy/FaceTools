import bpy
import sys
import os
import math
from mathutils import Vector, Matrix


def main(input_path):
    print("=== Начало выполнения скрипта ===")

    # Проверка входного пути
    print(f"[INFO] Входной файл: {input_path}")
    if not os.path.exists(input_path):
        print("[ERROR] Файл не найден!")
        return

    # Очистка сцены
    print("[INFO] Очистка сцены...")
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Импорт GLB
    print("[INFO] Импорт GLB-файла...")
    try:
        bpy.ops.import_scene.gltf(filepath=input_path)
    except Exception as e:
        print(f"[ERROR] Ошибка импорта GLB: {e}")
        return

    # Центрирование модели
    print("[INFO] Центрирование модели...")
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    if not mesh_objects:
        print("[ERROR] Не найдено объектов типа MESH после импорта GLB.")
        return

    # Расчет bounding box
    all_coords = []
    for obj in mesh_objects:
        matrix = obj.matrix_world
        for v in obj.data.vertices:
            all_coords.append(matrix @ v.co)

    if all_coords:
        min_co = Vector(map(min, zip(*all_coords)))
        max_co = Vector(map(max, zip(*all_coords)))
        center = (min_co + max_co) / 2
        size = max_co - min_co
        max_dim = max(size.x, size.y, size.z)
        print(f"[DEBUG] Размер модели: {size}, Центр: {center}, Макс. размер: {max_dim}")
    else:
        center = Vector((0, 0, 0))
        max_dim = 1.0
        print("[DEBUG] Не удалось определить размеры модели, используем значения по умолчанию")

    # Создание пустышки для вращения
    print("[INFO] Создание точки вращения...")
    bpy.ops.object.empty_add(type='PLAIN_AXES', radius=0.1)
    pivot = bpy.context.object
    pivot.name = "Rotation_Pivot"
    pivot.location = center

    # Связываем все объекты модели с пустышкой
    for obj in mesh_objects:
        obj.parent = pivot
        obj.matrix_parent_inverse = Matrix.Identity(4)

    # Масштабирование модели
    if max_dim > 0:
        scale_factor = max(0.7 / max_dim, 0.5)
        print(f"[DEBUG] Коэффициент масштабирования: {scale_factor}")
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objects:
            obj.select_set(True)
        bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))
        bpy.ops.object.transform_apply(scale=True)
        bpy.ops.object.select_all(action='DESELECT')
    else:
        print("[WARNING] Не удалось определить размер модели для масштабирования")

    # Настройка камеры
    print("[INFO] Настройка камеры...")
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = 'PERSP'
    camera.data.lens = 50

    # Оптимальное расстояние камеры
    distance = max_dim * 3  # Более реалистичный расчет
    camera.location = (0, -distance, 0)

    # Направление камеры
    direction = (center - camera.location).normalized()
    rot_quat = direction.to_track_quat('Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    # Легкий наклон вниз
    camera.rotation_euler.x += math.radians(10)

    print(f"[DEBUG] Позиция камеры: {camera.location}, Расстояние: {distance}")
    bpy.context.scene.camera = camera

    # Удаление старого света
    print("[INFO] Удаление старого света...")
    bpy.ops.object.select_all(action='DESELECT')
    for light in [o for o in bpy.context.scene.objects if o.type == 'LIGHT']:
        light.select_set(True)
    bpy.ops.object.delete()

    # Фоновое освещение
    print("[INFO] Настройка фонового освещения...")
    if not bpy.context.scene.world:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()

    output = nodes.new('ShaderNodeOutputWorld')
    background = nodes.new('ShaderNodeBackground')
    background.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1)  # Светло-серый
    background.inputs['Strength'].default_value = 1.5  # Уменьшено

    world.node_tree.links.new(background.outputs['Background'], output.inputs['Surface'])

    # Добавляем точечный свет для деталей
    bpy.ops.object.light_add(type='SUN', radius=0.5)
    sun = bpy.context.object
    sun.data.energy = 5.0
    sun.rotation_euler = (math.radians(45), math.radians(45), 0)

    # Настройка рендера
    print("[INFO] Настройка рендера...")
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    bpy.context.scene.render.resolution_x = 2048
    bpy.context.scene.render.resolution_y = 2048
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.context.scene.render.image_settings.quality = 90

    # Материалы
    print("[INFO] Применение материалов...")
    for obj in mesh_objects:
        if not obj.data.materials:
            mat = bpy.data.materials.new(name="Debug_Material")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            nodes.clear()

            emission = nodes.new('ShaderNodeEmission')
            emission.inputs['Color'].default_value = (1, 1, 1, 1)
            emission.inputs['Strength'].default_value = 1.0  # Уменьшено

            output = nodes.new('ShaderNodeOutputMaterial')
            mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])

            obj.data.materials.append(mat)
        else:
            for mat_slot in obj.material_slots:
                if mat_slot.material:
                    mat = mat_slot.material
                    mat.use_nodes = True
                    nodes = mat.node_tree.nodes
                    nodes.clear()

                    emission = nodes.new('ShaderNodeEmission')
                    emission.inputs['Color'].default_value = (1, 1, 1, 1)
                    emission.inputs['Strength'].default_value = 0.8

                    output = nodes.new('ShaderNodeOutputMaterial')
                    mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])

    # Папка для рендеров
    output_dir = os.path.join(os.path.dirname(input_path), "renders")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Папка для рендеров: {output_dir}")

    # Рендеринг
    steps = 12
    print(f"[INFO] Рендеринг {steps} углов...")
    for i in range(steps):
        angle = (360 / steps) * i
        print(f"[INFO] Угол {angle} градусов...")

        pivot.rotation_euler = (0, 0, -math.radians(angle))
        bpy.context.view_layer.update()

        file_name = f"{os.path.splitext(os.path.basename(input_path))[0]}_{int(angle):03d}deg.jpg"
        bpy.context.scene.render.filepath = os.path.join(output_dir, file_name)

        try:
            print(f"[DEBUG] Сохранение в: {bpy.context.scene.render.filepath}")
            bpy.ops.render.render(write_still=True)
            print(f"[INFO] Сохранено: {bpy.context.scene.render.filepath}")
        except Exception as e:
            print(f"[ERROR] Ошибка рендеринга: {e}")
            return

    print("=== Скрипт завершен ===")


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    if not args:
        print("[ERROR] Не указан входной файл!")
        sys.exit(1)

    input_path = args[0]
    main(input_path)