import insightface
import numpy as np
import cv2
from pathlib import Path
import shutil
import logging
from datetime import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Удаление дубликатов лиц (оставляется изображение с большей площадью)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Примеры использования:
  # Анализ (без удаления):
  python arcface_cleanup.py /path/to/faces /path/to/unique

  # Реальное удаление дубликатов:
  python arcface_cleanup.py /path/to/faces /path/to/unique --threshold 0.85 --delete
        '''
    )
    parser.add_argument('input_dir', type=str, help='Директория с исходными изображениями')
    parser.add_argument('output_dir', type=str, help='Директория для уникальных лиц')
    parser.add_argument('--threshold', '-t', type=float, default=0.7,
                        help='Порог уникальности 0-1 (default: 0.7, чем выше - строже)')
    parser.add_argument('--det-size', '-d', type=int, default=640,
                        help='Размер для детекции лиц (default: 640px)')
    parser.add_argument('--delete', action='store_true',
                        help='⚠️  ВКЛЮЧИТЬ реальное удаление файлов (по умолчанию - dry-run)')
    return parser.parse_args()

# === НАСТРОЙКА ЛОГИРОВАНИЯ ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('arcface_cleanup.log', encoding='utf-8')
    ]
)
log = logging.getLogger(__name__)

# === ПАРАМЕТРЫ ===
args = parse_args()
input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
threshold = args.threshold
output_dir.mkdir(exist_ok=True)
det_size = (640, 640)

# БЕЗОПАСНЫЙ РЕЖИМ: True = только логирование, False = реальное удаление
DRY_RUN = False  # ← ВНИМАНИЕ: установите False для реального удаления!

# === СТАТИСТИКА ===
stats = {
    'total_files': 0,
    'not_images': 0,
    'no_faces': [],
    'processed': 0,
    'unique_kept': [],
    'replaced_by_size': [],  # Замененные (меньшие удалены)
    'deleted_files': []  # Список удаленных файлов
}


# === ФУНКЦИИ ===
def get_image_area(img_path):
    """Возвращает площадь изображения в пикселях"""
    try:
        img = cv2.imread(str(img_path))
        if img is not None:
            return img.shape[0] * img.shape[1]  # height * width
    except Exception as e:
        log.error(f"Ошибка подсчета площади для {img_path}: {e}")
    return 0


def safe_delete(file_path, dry_run=DRY_RUN):
    """Безопасное удаление файла с логированием"""
    if dry_run:
        log.warning(f"  🚧 DRY-RUN: бы удален {file_path}")
        stats['deleted_files'].append(f"{file_path} (DRY-RUN)")
        return True

    try:
        file_path.unlink()
        log.info(f"  🗑️  Файл удален: {file_path}")
        stats['deleted_files'].append(str(file_path))
        return True
    except Exception as e:
        log.error(f"  ❌ Ошибка удаления {file_path}: {e}")
        return False


# === ИНИЦИАЛИЗАЦИЯ ===
log.info('=' * 80)
log.info(f"🚀 СТАРТ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info(f"📂 Входная директория: {input_dir}")
log.info(f"💾 Выходная директория: {output_dir}")
log.info(f"📊 Порог уникальности: {threshold}")
log.info(f"🧪 РЕЖИМ: {'ТОЛЬКО ЛОГИРОВАНИЕ' if DRY_RUN else 'РЕАЛЬНОЕ УДАЛЕНИЕ ФАЙЛОВ'}")
log.info('=' * 80)

# Проверка директории
if not input_dir.exists():
    log.error(f'❌ Директория не существует: {input_dir}')
    exit(1)

# Загрузка моделей
log.info("Загрузка моделей InsightFace...")
try:
    detector = insightface.app.FaceAnalysis()
    detector.prepare(ctx_id=-1, det_size=det_size)
    log.info("✅ Модели успешно загружены")
except Exception as e:
    log.error(f'❌ Ошибка загрузки моделей: {e}', exc_info=True)
    exit(1)

# === ПРОЦЕССИНГ ===
all_files = list(input_dir.glob('*'))
valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.heic'}
stats['total_files'] = len(all_files)

log.info(f'📝 Найдено файлов: {stats["total_files"]}')
log.debug(f'Допустимые расширения: {valid_extensions}')
log.info('=' * 80)

embeddings = []
filepaths = []

for idx, img_path in enumerate(all_files, 1):
    log.info(f'[{idx:>3}/{len(all_files)}] Обрабатывается: {img_path.name}')

    # Валидация файла
    if img_path.suffix.lower() not in valid_extensions:
        log.debug(f'  ↳ Пропущено: недопустимое расширение')
        stats['not_images'] += 1
        continue

    if img_path.stat().st_size == 0:
        log.warning(f'  ↳ Пропущено: файл пустой')
        stats['not_images'] += 1
        continue

    # Чтение изображения
    img = cv2.imread(str(img_path))
    if img is None:
        log.error(f'  ↳ ОШИБКА: невозможно прочитать изображение')
        stats['not_images'] += 1
        continue

    log.debug(f'  ↳ Размер: {img.shape[1]}x{img.shape[0]}px, Каналы: {img.shape[2]}')

    # Детекция лиц
    try:
        faces = detector.get(img)
    except Exception as e:
        log.error(f'  ↳ ОШИБКА детекции: {e}', exc_info=True)
        continue

    if not faces:
        log.warning(f'  ↳ Лицо НЕ ОБНАРУЖЕНО')
        stats['no_faces'].append({
            'file': img_path.name,
            'size': f'{img.shape[1]}x{img.shape[0]}'
        })
        continue

    logging.info(f'  ↳ Найдено лиц: {len(faces)}')
    for f_idx, face in enumerate(faces, 1):
        log.debug(f'    - Лицо #{f_idx}: score={face.det_score:.3f}, '
                  f'пол={face.gender}, возраст={face.age}, '
                  f'bbox={face.bbox}')

    # Выбираем лучшее лицо
    best_face = max(faces, key=lambda f: f.det_score)
    if best_face.det_score < 0.5:
        log.warning(f'  ↳ Лицо найдено, но низкая уверенность: {best_face.det_score:.3f}')

    embeddings.append(best_face.normed_embedding)
    filepaths.append(img_path)
    stats['processed'] += 1
    log.info(f'  ✅ Эмбеддинг сохранен (score={best_face.det_score:.3f})')

# === АНАЛИЗ УНИКАЛЬНОСТИ С УДАЛЕНИЕМ ДУБЛИКАТОВ ===
log.info('=' * 80)
log.info(f"СТАТУС: Обработано {stats['processed']} лиц")
if stats['no_faces']:
    log.warning(f"Лица не найдены: {len(stats['no_faces'])} файлов")
if stats['not_images']:
    log.warning(f"Пропущено (не изображения): {stats['not_images']}")
log.info('=' * 80)

if stats['processed'] == 0:
    log.error('❌ КРИТИЧЕСКАЯ ОШИБКА: Нет изображений с лицами!')
    exit(1)

# Кластеризация с удалением меньших дубликатов
log.info('=' * 80)
log.info(f'🔍 Начало кластеризации с удалением дубликатов (порог={threshold})...')
log.info('=' * 80)

embeddings = np.array(embeddings)  # (N, 512)
unique_indices = []
# Список для хранения эмбеддингов уникальных лиц (для удобства)
unique_embeddings = []

for i in range(embeddings.shape[0]):
    current_emb = embeddings[i:i + 1]  # (1, 512)
    current_path = filepaths[i]

    if len(unique_indices) == 0:
        # Первый элемент всегда добавляется
        unique_indices.append(i)
        unique_embeddings.append(current_emb)
        stats['unique_kept'].append(current_path.name)
        log.info(f'  Шаг {i + 1:>3}: {current_path.name} -> ДОБАВЛЕН (первый)')
        continue

    # Сравнение с уже отобранными
    selected_embs = np.concatenate(unique_embeddings, axis=0)  # (k, 512)
    sims = np.dot(current_emb, selected_embs.T).flatten()  # (k,)
    max_sim = sims.max()
    best_match_local_idx = sims.argmax()
    best_match_global_idx = unique_indices[best_match_local_idx]
    best_match_path = filepaths[best_match_global_idx]

    # Получаем площади для сравнения
    current_area = get_image_area(current_path)
    best_match_area = get_image_area(best_match_path)

    log.info(f'  Шаг {i + 1:>3}: {current_path.name} (area={current_area:,}px²) '
             f'| max_sim={max_sim:.3f} к {best_match_path.name} (area={best_match_area:,}px²)')

    if max_sim < threshold:
        # Уникальное лицо
        unique_indices.append(i)
        unique_embeddings.append(current_emb)
        stats['unique_kept'].append(current_path.name)
        log.info(f'    -> ✅ ДОБАВЛЕН (уникальный)')
    else:
        # Дубликат — сравниваем размеры
        if current_area > best_match_area:
            # Текущий файл больше — заменяем и удаляем старый
            log.info(f'    -> 🔄 ЗАМЕНА: текущий больше ({current_area:,} > {best_match_area:,} px²)')

            # Удаляем старый файл
            safe_delete(best_match_path, DRY_RUN)

            # Заменяем в списках
            unique_indices[best_match_local_idx] = i
            unique_embeddings[best_match_local_idx] = current_emb
            stats['replaced_by_size'].append({
                'deleted': best_match_path.name,
                'kept': current_path.name,
                'reason': f'больше на {current_area - best_match_area:,} px²',
                'similarity': float(max_sim)
            })
            # Убираем старый из kept, добавляем новый
            if best_match_path.name in stats['unique_kept']:
                stats['unique_kept'].remove(best_match_path.name)
            stats['unique_kept'].append(current_path.name)
        else:
            # Текущий файл меньше или равен — удаляем текущий
            log.info(f'    -> 🗑️  УДАЛЕНИЕ: текущий меньше ({current_area:,} <= {best_match_area:,} px²)')
            safe_delete(current_path, DRY_RUN)

# === ИТОГИ ===
log.info('=' * 80)
log.info('📊 ИТОГОВЫЙ ОТЧЕТ')
log.info('=' * 80)
log.info(f'✅ Уникальных лиц в итоге: {len(unique_indices)}')
log.info(f'🔄 Заменено по размеру: {len(stats["replaced_by_size"])}')
log.info(f'🗑️  Всего удалено файлов: {len(stats["deleted_files"])}')
log.info(f'📂 Всего файлов: {stats["total_files"]}')
log.info(f'📷 Обработано лиц: {stats["processed"]}')
log.info(f'⚠️  Пропущено: {stats["not_images"] + len(stats["no_faces"])}')

# Сохранение уникальных файлов (если они еще не в output_dir)
log.info('=' * 80)
log.info('💾 Копирование уникальных файлов (если нужно)...')
for idx in unique_indices:
    src = filepaths[idx]
    dst = output_dir / src.name
    if not dst.exists():  # Копируем только если нет
        try:
            shutil.copy2(src, dst)
            log.debug(f'  ✓ {src.name}')
        except Exception as e:
            log.error(f'  ✗ Ошибка копирования {src.name}: {e}')

log.info(f'📁 Уникальные файлы в: {output_dir}')

# === ДЕТАЛЬНЫЕ ОТЧЕТЫ ===
if stats['no_faces']:
    log.info('\n' + '=' * 80)
    log.info('🔍 ФАЙЛЫ БЕЗ ОБНАРУЖЕННЫХ ЛИЦ (топ-10):')
    log.info('=' * 80)
    for item in stats['no_faces'][:10]:
        log.warning(f"  - {item['file']} ({item['size']})")

if stats['replaced_by_size']:
    log.info('\n' + '=' * 80)
    log.info('🔄 ЗАМЕНЕНО ПО РАЗМЕРУ (удалено -> сохранено):')
    log.info('=' * 80)
    for repl in sorted(stats['replaced_by_size'], key=lambda x: x['similarity'], reverse=True):
        log.info(f"  - {repl['deleted']:<30} -> {repl['kept']:<30}")
        log.info(f"    Причина: {repl['reason']}, Сходство: {repl['similarity']:.3f}")

if stats['deleted_files']:
    log.info('\n' + '=' * 80)
    log.info(f'🗑️  СПИСОК УДАЛЕННЫХ ФАЙЛОВ ({len(stats["deleted_files"])} шт):')
    log.info('=' * 80)
    for deleted_path in stats['deleted_files'][:20]:  # Показать первые 20
        log.warning(f"  - {deleted_path}")
    if len(stats['deleted_files']) > 20:
        log.info(f'  ... и еще {len(stats["deleted_files"]) - 20} файлов')

log.info('\n' + '=' * 80)
log.info('✨ ГОТОВО!')
log.info(f'📂 Проверьте папку: {output_dir}')
log.info(f'📄 Полный лог: arcface_cleanup.log')
if DRY_RUN:
    log.warning('⚠️  БЫЛ ВКЛЮЧЕН РЕЖИМ DRY-RUN, ФАЙЛЫ НЕ УДАЛЯЛИСЬ!')
log.info('=' * 80)