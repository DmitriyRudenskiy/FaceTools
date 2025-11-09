### Обновленная систематизированная документация с дополнениями

---

#### **1. Установка зависимостей**  
```bash
pip install mypy ruff pylint black isort autoflake docformatter
```

---

#### **2. Основные этапы рабочего процесса**  
**Рекомендуемый порядок** (оптимизирован для избежания конфликтов):  
1. **Исправление синтаксиса** → `ruff` (автоматическое исправление ошибок).  
2. **Форматирование кода** → `black` + `isort`.  
3. **Удаление мусора** → `autoflake`.  
4. **Форматирование docstrings** → `docformatter`.  
5. **Проверка типов** → `mypy`.  
6. **Глубокий линтинг** → `pylint`.  

---

#### **3. Детальные команды с обработкой сложных путей**  
*Используйте версию с `find ... -print0 | xargs -0`, если в проекте есть файлы с пробелами в именах.*

##### **3.1. Исправление синтаксиса и стиля**  
- **Автоматическое исправление через `ruff`** (рекомендуется):  
  ```bash
  ruff check --fix . --exclude .venv
  ```
  *Альтернатива с обработкой пробелов в путях:*  
  ```bash
  find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 ruff check --fix
  ```

##### **3.2. Форматирование кода**  
- **Через `black`** (строгое форматирование):  
  ```bash
  black . --exclude .venv
  ```
  *Альтернатива:*  
  ```bash
  find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 black
  ```

- **Сортировка импортов** (с профилем `black`):  
  ```bash
  isort . --profile black --skip .venv
  ```
  *Альтернатива:*  
  ```bash
  find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 isort
  ```

##### **3.3. Удаление неиспользуемого кода**  
```bash
find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 autoflake \
  --in-place --remove-all-unused-imports --remove-unused-variables
```
> **Примечание**: Запускайте **после** форматирования (`black`/`isort`), чтобы избежать случайного удаления импортов.

##### **3.4. Форматирование docstrings**  
```bash
find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 docformatter \
  --in-place --wrap-summaries=100 --wrap-description=100
```

##### **3.5. Проверка типов**  
```bash
mypy . --exclude ".venv" --show-error-codes --pretty
```

##### **3.6. Глубокий линтинг**  
- **Строгая проверка**:  
  ```bash
  find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 pylint
  ```
- **Проверка с кастомной конфигурацией** (из `pyproject.toml`):  
  ```bash
  find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 pylint --rcfile=pyproject.toml
  ```

---

#### **4. Полезные скрипты**

##### **4.1. Полная автоматизация (с обработкой пробелов в путях)**  
```bash
#!/bin/bash
# 1. Исправить ошибки через ruff
find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 ruff check --fix

# 2. Удалить неиспользуемый код
find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 autoflake \
  --in-place --remove-all-unused-imports --remove-unused-variables

# 3. Форматировать код
find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 black
find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 isort

# 4. Форматировать docstrings
find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 docformatter --in-place

# 5. Проверить типы и линтинг
mypy . --exclude ".venv"
find . -type d -name ".venv" -prune -o -type f -name "*.py" -print0 | xargs -0 pylint --rcfile=pyproject.toml
```

##### **4.2. Pre-commit hook (рекомендуется)**  
Добавьте в `.pre-commit-config.yaml`:  
```yaml
repos:
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: ["--fix"]
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/myint/autoflake
    rev: v2.2.1
    hooks:
      - id: autoflake
        args: ["--in-place", "--remove-all-unused-imports", "--remove-unused-variables"]
```
Затем:  
```bash
pre-commit install
```

---

#### **5. Критически важные замечания**  
1. **Порядок шагов** — если запустить `autoflake` **до** `isort/black`, он может удалить импорты, которые будут добавлены позже.  
2. **Дублирование команд** — в исходных примерах были повторяющиеся вызовы `ruff`, `isort`, `autoflake`. В обновленной версии это устранено.  
3. **Обработка путей** — версии с `find ... -print0 | xargs -0` гарантируют корректную работу с файлами, содержащими пробелы или специальные символы.  
4. **Замена `autopep8`** — `ruff` покрывает 90% задач `autopep8` + `pylint --fix`, поэтому отдельный запуск `autopep8` не требуется.  

---

#### **6. Если нужно минимальное решение**  
Используйте только `ruff` и `black`:  
```bash
# Исправить ошибки и отформатировать
ruff check --fix . --exclude .venv
black . --exclude .venv
```