import os
import json
import cv2
import argparse
import numpy as np
from mmpose.apis import init_model, inference_top_down_pose_model
from mmdet.apis import init_detector, inference_detector
from mmpose.datasets import DatasetInfo


def export_face_keypoints_to_json(image_path, output_json_path,
                                  det_config='configs/yolox-s_8xb8-300e_coco-face.py',
                                  det_checkpoint='checkpoints/yolo-x_8xb8-300e_coco-face_13274d7c.pth',
                                  pose_config='configs/rtmpose-m_8xb256-120e_face6-256x256.py',
                                  pose_checkpoint='checkpoints/rtmpose-m_simcc-face6_pt-in1k_120e-256x256.pth',
                                  device='cpu'):
    """
    Экспорт ключевых точек лица в JSON файл с использованием MMDetection + RTMPose

    Args:
        image_path: Путь к изображению
        output_json_path: Путь для сохранения JSON файла
        det_config: Конфигурация детектора лиц
        det_checkpoint: Веса детектора лиц
        pose_config: Конфигурация модели позы
        pose_checkpoint: Веса модели позы
        device: Устройство для вычислений ('cpu')
    """

    # Проверка существования файла изображения
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    # Проверка существования файлов моделей
    model_files = {
        "detector config": det_config,
        "detector weights": det_checkpoint,
        "pose config": pose_config,
        "pose weights": pose_checkpoint
    }

    missing_files = []
    for name, path in model_files.items():
        if not os.path.exists(path) and not path.startswith('http'):
            missing_files.append(f"{name} ({path})")

    if missing_files:
        print("ПРЕДУПРЕЖДЕНИЕ: Не найдены следующие локальные файлы:")
        for file in missing_files:
            print(f"  - {file}")
        print("Попытка использования онлайн-ресурсов...\n")

    # Инициализация детектора лиц
    print(f"Загрузка детектора лиц (YOLOX) из {det_config}...")
    det_model = init_detector(det_config, det_checkpoint, device=device)

    # Инициализация модели ключевых точек
    print(f"Загрузка модели ключевых точек (RTMPose) из {pose_config}...")
    pose_model = init_model(pose_config, pose_checkpoint, device=device)

    # Загрузка информации о датасете
    dataset_info = pose_model.cfg.data.get('test', {}).get('dataset_info', None)
    if dataset_info:
        dataset_info = DatasetInfo(dataset_info)

    # Обнаружение лиц с помощью MMDetection
    print("Обнаружение лиц на изображении...")
    mmdet_results = inference_detector(det_model, image_path)

    # Фильтрация результатов (класс лиц = 0 в COCO)
    person_results = []
    for bbox in mmdet_results[0]:
        if bbox[4] > 0.5:  # Порог уверенности
            person = {}
            person['bbox'] = bbox
            person_results.append(person)

    if not person_results:
        print("Лица не обнаружены на изображении")
        # Создаем пустой JSON файл
        with open(output_json_path, 'w') as f:
            json.dump([], f)
        return []

    # Получение ключевых точек для обнаруженных лиц
    print(f"Обнаружено {len(person_results)} лиц. Получение ключевых точек...")
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        image_path,
        person_results,
        bbox_thr=0.5,
        dataset_info=dataset_info
    )

    # Подготовка результатов для JSON
    results = []
    for person_id, res in enumerate(pose_results):
        # Формат keypoints: [x1, y1, score1, x2, y2, score2, ...]
        keypoints_flat = res['keypoints'].tolist()

        # Преобразуем в структурированный формат [[x, y, score], ...]
        keypoints_structured = []
        for i in range(0, len(keypoints_flat), 3):
            if i + 2 < len(keypoints_flat):
                keypoints_structured.append([
                    float(keypoints_flat[i]),
                    float(keypoints_flat[i + 1]),
                    float(keypoints_flat[i + 2])
                ])

        # Получаем bounding box
        bbox = res['bbox'].tolist()

        # Получаем размеры изображения
        img = cv2.imread(image_path)
        height, width = img.shape[:2] if img is not None else [0, 0]

        # Определяем имена ключевых точек (если доступно)
        keypoint_names = []
        if dataset_info and hasattr(dataset_info, 'keypoint_names'):
            keypoint_names = dataset_info.keypoint_names
        else:
            keypoint_names = [f"point_{i}" for i in range(len(keypoints_structured))]

        results.append({
            "image_id": os.path.basename(image_path),
            "image_width": int(width),
            "image_height": int(height),
            "person_id": person_id,
            "bbox": [float(x) for x in bbox],
            "num_keypoints": len(keypoints_structured),
            "keypoints": keypoints_structured,
            "keypoint_names": keypoint_names,
            "detection_score": float(bbox[4]),  # Скор из детектора
            "model_type": "rtmpose-m_face6"
        })

    # Сохранение в JSON
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nРезультаты сохранены в {output_json_path}")
    print(f"Обнаружено лиц: {len(pose_results)}")
    if pose_results:
        print(f"Количество ключевых точек на лицо: {len(keypoints_structured)}")
        print(f"Тип ключевых точек: {keypoint_names}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Экспорт ключевых точек лица в JSON')

    # Обязательные аргументы
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Путь к входному изображению')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Путь для сохранения JSON файла (по умолчанию: input_keypoints.json)')

    # Аргументы для детектора
    parser.add_argument('--det-config', type=str,
                        default='configs/yolox-s_8xb8-300e_coco-face.py',
                        help='Конфигурация детектора лиц')
    parser.add_argument('--det-checkpoint', type=str,
                        default='checkpoints/yolo-x_8xb8-300e_coco-face_13274d7c.pth',
                        help='Веса детектора лиц')

    # Аргументы для модели позы
    parser.add_argument('--pose-config', type=str,
                        default='configs/rtmpose-m_8xb256-120e_face6-256x256.py',
                        help='Конфигурация модели позы')
    parser.add_argument('--pose-checkpoint', type=str,
                        default='checkpoints/rtmpose-m_simcc-face6_pt-in1k_120e-256x256.pth',
                        help='Веса модели позы')

    # Дополнительные параметры
    parser.add_argument('--device', type=str, default='cpu',
                        help='Устройство для вычислений (cpu или cuda:0)')
    parser.add_argument('--bbox-thr', type=float, default=0.5,
                        help='Порог уверенности для обнаружения лиц')

    args = parser.parse_args()

    # Определяем имя выходного файла, если не указано
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{base_name}_keypoints.json"

    # Запускаем экспорт
    try:
        export_face_keypoints_to_json(
            image_path=args.input,
            output_json_path=args.output,
            det_config=args.det_config,
            det_checkpoint=args.det_checkpoint,
            pose_config=args.pose_config,
            pose_checkpoint=args.pose_checkpoint,
            device=args.device
        )
    except Exception as e:
        print(f"Ошибка при обработке: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()