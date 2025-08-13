# Базовый запуск
#blender -b -P render_skin_textures.py -- model.stl

# С текстурой и пользовательской директорией
#blender -b -P render_skin_textures.py -- model.stl -t skin_texture.jpg -o my_renders

# С шагом 30 градусов
#blender -b -P render_skin_textures.py -- model.stl -a 30

# Полная команда
#blender -b -P render_skin_textures.py -- /home/user/model.stl -t /home/user/skin.jpg -o /home/user/results -a 15

import bpy
import os
import math
import sys
import argparse


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for material in bpy.data.materials:
        bpy.data.materials.remove(material)

    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture)


def load_stl_model(filepath):
    bpy.ops.import_mesh.stl(filepath=filepath)
    return bpy.context.active_object


def create_skin_material():
    material = bpy.data.materials.new(name="SkinMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    nodes.clear()

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    texture_node = nodes.new(type='ShaderNodeTexImage')

    output_node.location = (400, 0)
    principled_node.location = (200, 0)
    texture_node.location = (-200, 0)

    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    links.new(texture_node.outputs['Color'], principled_node.inputs['Base Color'])

    principled_node.inputs['Roughness'].default_value = 0.8
    principled_node.inputs['Specular'].default_value = 0.3
    principled_node.inputs['Subsurface'].default_value = 0.1
    principled_node.inputs['Subsurface Radius'].default_value = (1.0, 0.2, 0.1)

    return material, texture_node


def load_skin_texture(texture_node, texture_path=None):
    if texture_path and os.path.exists(texture_path):
        try:
            image = bpy.data.images.load(texture_path)
            texture_node.image = image
            print(f"Текстура загружена: {texture_path}")
        except Exception as e:
            print(f"Ошибка загрузки текстуры: {e}")
            create_procedural_texture(texture_node)
    else:
        create_procedural_texture(texture_node)


def create_procedural_texture(texture_node):
    nodes = texture_node.id_data.nodes
    links = texture_node.id_data.links

    noise_node1 = nodes.new(type='ShaderNodeTexNoise')
    noise_node2 = nodes.new(type='ShaderNodeTexNoise')
    mix_node = nodes.new(type='ShaderNodeMixRGB')
    color_ramp_node = nodes.new(type='ShaderNodeValToRGB')

    noise_node1.location = (-600, 100)
    noise_node2.location = (-600, -100)
    mix_node.location = (-400, 0)
    color_ramp_node.location = (-200, 0)

    noise_node1.inputs['Scale'].default_value = 10.0
    noise_node1.inputs['Detail'].default_value = 2.0
    noise_node2.inputs['Scale'].default_value = 50.0
    noise_node2.inputs['Detail'].default_value = 4.0

    mix_node.inputs['Fac'].default_value = 0.5

    color_ramp_node.color_ramp.elements[0].color = (0.8, 0.6, 0.4, 1)
    color_ramp_node.color_ramp.elements[1].color = (0.6, 0.4, 0.3, 1)

    links.new(noise_node1.outputs['Fac'], mix_node.inputs[1])
    links.new(noise_node2.outputs['Fac'], mix_node.inputs[2])
    links.new(mix_node.outputs['Color'], color_ramp_node.inputs['Fac'])
    links.new(color_ramp_node.outputs['Color'], texture_node.inputs['Color'])

    print("Создана процедурная текстура кожи")


def setup_camera():
    bpy.ops.object.camera_add(location=(0, -5, 0))
    camera = bpy.context.active_object
    camera.rotation_euler = (math.radians(90), 0, 0)

    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 3.0

    bpy.context.scene.camera = camera
    return camera


def setup_lighting():
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()

    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 3.0

    bpy.ops.object.light_add(type='AREA', location=(-3, -3, 5))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 100.0
    fill_light.data.size = 2.0


def render_rotations(obj, output_dir, angle_step=15):
    os.makedirs(output_dir, exist_ok=True)

    original_rotation = obj.rotation_euler.copy()

    scene = bpy.context.scene
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'

    total_renders = 360 // angle_step
    print(f"Будет создано {total_renders} рендеров")

    for i, angle in enumerate(range(0, 360, angle_step)):
        obj.rotation_euler[2] = math.radians(angle)

        filepath = os.path.join(output_dir, f"render_{angle:03d}.png")
        scene.render.filepath = filepath

        bpy.ops.render.render(write_still=True)

        progress = (i + 1) / total_renders * 100
        print(f"[{progress:3.0f}%] Создан рендер: {filepath}")

    obj.rotation_euler = original_rotation


def main(stl_path, texture_path=None, output_dir="renders", angle_step=15):
    try:
        print("Начало обработки...")
        print(f"STL файл: {stl_path}")
        print(f"Текстура: {texture_path or 'Процедурная'}")
        print(f"Выходная директория: {output_dir}")

        if not os.path.exists(stl_path):
            raise FileNotFoundError(f"STL файл не найден: {stl_path}")

        clear_scene()

        print("Загрузка STL модели...")
        obj = load_stl_model(stl_path)

        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        obj.location = (0, 0, 0)

        print("Создание материала кожи...")
        material, texture_node = create_skin_material()
        obj.data.materials.append(material)

        print("Загрузка текстуры...")
        load_skin_texture(texture_node, texture_path)

        print("Настройка камеры и освещения...")
        setup_camera()
        setup_lighting()

        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 128

        print("Создание рендеров...")
        render_rotations(obj, output_dir, angle_step)

        print("Все рендеры созданы успешно!")

    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Render STL model with skin texture from multiple angles')
    parser.add_argument('stl_file', help='Path to STL file')
    parser.add_argument('-t', '--texture', help='Path to skin texture image (optional)')
    parser.add_argument('-o', '--output', default='renders', help='Output directory for renders')
    parser.add_argument('-a', '--angle', type=int, default=15, help='Angle step in degrees (default: 15)')

    # Если скрипт запущен из Blender, sys.argv будет отличаться
    if len(sys.argv) > 4:  # Blender передает свои аргументы
        # Извлекаем пользовательские аргументы
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
        if not argv:
            argv = sys.argv[-4:]  # Предполагаем последние 4 аргумента

        args = parser.parse_args(argv)
    else:
        args = parser.parse_args()

    main(args.stl_file, args.texture, args.output, args.angle)