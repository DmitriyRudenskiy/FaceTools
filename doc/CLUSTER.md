# Варианты запуска кластеризации лиц

На основе структуры проекта и реализованных алгоритмов, вот все возможные варианты запуска с различными параметрами:

## 1. Базовые варианты запуска

### Запуск с методом по умолчанию (ATCC)
```bash
python /Users/user/PycharmProjects/FaceTools/cli/group_files.py --json matrix.json
```
- Использует метод ATCC с параметрами по умолчанию
- Сохраняет результат в groups.json
- Организует файлы в папку groups

### Запуск с указанием выходного файла
```bash
python cli/group_files.py --json matrix.json --out results.json
```
- Результат сохраняется в results.json вместо groups.json

### Запуск без организации файлов
```bash
python cli/group_files.py --json matrix.json --dest ""
```
- Обрабатывает матрицу, но не копирует файлы в отдельные каталоги

## 2. Варианты с разными методами кластеризации

### Иерархическая кластеризация с настройками
```bash
python cli/group_files.py --json matrix.json --method hierarchical \
                          --linkage complete \
                          --inconsistency-threshold 1.2 \
                          --min-cluster-size 3
```
- Использует метод полной связности
- Более строгий порог несогласованности (1.2 вместо 1.5)
- Минимальный размер кластера 3

### Спектральная кластеризация
```bash
python cli/group_files.py --json matrix.json --method spectral \
                          --min-cluster-size 2
```
- Автоматически определяет количество кластеров через eigengap
- Минимальный размер кластера 2

### DBSCAN с ручным параметром eps
```bash
python cli/group_files.py --json matrix.json --method dbscan \
                          --eps 0.3 \
                          --min-cluster-size 2
```
- Использует ручное значение eps = 0.3
- Минимальный размер кластера 2
- Не будет автоматически определять eps

### DBSCAN с автоматическим определением eps
```bash
python cli/group_files.py --json matrix.json --method dbscan \
                          --min-cluster-size 3
```
- eps будет автоматически определен через анализ k-расстояний
- Использует k = min-cluster-size = 3

### DBSCAN с настройкой k_for_eps
```bash
python cli/group_files.py --json matrix.json --method dbscan \
                          --min-cluster-size 2 \
                          --k-for-eps 5
```
- Для определения eps использует 5-е ближайшее расстояние
- Минимальный размер кластера 2

## 3. Комбинации параметров для сложных сценариев

### Иерархическая кластеризация с минимальным размером кластера 1
```bash
python cli/group_files.py --json matrix.json --method hierarchical \
                          --min-cluster-size 1
```
- Позволяет создавать одиночные кластеры (если они проходят проверку на несогласованность)

### Сравнение результатов разных методов
```bash
# Сохраняем результаты ATCC
python cli/group_files.py --json matrix.json --method atcc --out atcc_results.json

# Сохраняем результаты Hierarchical
python cli/group_files.py --json matrix.json --method hierarchical --out hierarchical_results.json

# Сохраняем результаты DBSCAN
python cli/group_files.py --json matrix.json --method dbscan --out dbscan_results.json
```
- Позволяет сравнить результаты разных алгоритмов на одних и тех же данных

### Запуск с высоким уровнем детализации
```bash
python cli/group_files.py --json matrix.json --method hierarchical \
                          --linkage average \
                          --inconsistency-threshold 2.0 \
                          --min-cluster-size 2
```
- Более высокий порог несогласованности (2.0) позволяет создавать больше кластеров
- Подходит для данных с большим количеством схожих лиц

### Запуск с минимальным количеством кластеров
```bash
python cli/group_files.py --json matrix.json --method hierarchical \
                          --inconsistency-threshold 0.5 \
                          --min-cluster-size 3
```
- Низкий порог несогласованности (0.5) приведет к меньшему количеству кластеров
- Подходит для данных с низким разнообразием лиц

## 4. Специальные сценарии

### Запуск с матрицей в другом формате
```bash
python cli/group_files.py --json custom_matrix.json \
                          --method spectral \
                          --min-cluster-size 2
```
- Использует пользовательский JSON-файл с матрицей схожести

### Запуск только для анализа без сохранения файлов
```bash
python cli/group_files.py --json matrix.json --dest "" --out analysis.json
```
- Сохраняет только JSON-результат без копирования файлов

### Запуск с высоким минимальным размером кластера
```bash
python cli/group_files.py --json matrix.json --method atcc \
                          --min-cluster-size 5
```
- Подходит для фильтрации шума и выбросов
- Будут созданы только крупные кластеры (5+ изображений)

## 5. Расширенные варианты

### Запуск с нестандартными параметрами для DBSCAN
```bash
python cli/group_files.py --json matrix.json --method dbscan \
                          --eps 0.4 \
                          --min-cluster-size 4
```
- Ручной eps = 0.4 (40% различия)
- Более высокий min_cluster_size = 4

### Запуск с иерархической кластеризацией и методом Уорда
```bash
python cli/group_files.py --json matrix.json --method hierarchical \
                          --linkage ward \
                          --min-cluster-size 2
```
- Использует метод Уорда, который минимизирует дисперсию внутри кластеров
- Подходит для данных с четко выраженными кластерами

## 6. Полный пример с максимальной настройкой
```bash
python cli/group_files.py --json matrix.json \
                          --method hierarchical \
                          --linkage complete \
                          --inconsistency-threshold 1.8 \
                          --min-cluster-size 3 \
                          --out detailed_results.json \
                          --dest organized_faces
```
- Использует метод полной связности
- Высокий порог несогласованности для более строгого разделения
- Минимальный размер кластера 3
- Сохраняет результаты в detailed_results.json
- Организует файлы в папку organized_faces

## Замечания по использованию

1. **Для ATCC** достаточно указать только `--min-cluster-size`, остальные параметры игнорируются
2. **Для Hierarchical** важны параметры `--linkage` и `--inconsistency-threshold`
3. **Для DBSCAN** если не указан `--eps`, он будет определен автоматически
4. **Для Spectral** можно добавить параметр `--max-clusters` для ограничения количества кластеров
5. Все параметры, не относящиеся к выбранному методу, игнорируются

Каждый метод имеет свои сильные стороны и подходит для разных типов данных, поэтому рекомендуется экспериментировать с параметрами для достижения наилучших результатов на конкретном наборе данных.