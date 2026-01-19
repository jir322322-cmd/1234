# Tile Stitcher (GUI, JPG → TIFF)

Настольное приложение для склейки большого изображения из набора JPG-тайлов с учётом:

- **белых полей по краям** (обрезаем только внешние белые рамки),
- **индивидуального наклона каждого тайла** (deskew до ±`max-angle`),
- **авто-перекрытия до 15 px** между соседями,
- **точной стыковки без щелей**,
- **вывода в lossless TIFF** (BigTIFF при необходимости).

Подходит для Windows 10/11, также кроссплатформенно.

## Требования

- Python 3.11+
- `opencv-python`
- `numpy`
- `Pillow`
- `PySide6`

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
# source .venv/bin/activate  # macOS/Linux
pip install --upgrade pip
pip install opencv-python numpy Pillow PySide6
```

### Windows (PowerShell): если не активируется venv

Если PowerShell пишет, что не удаётся загрузить модуль `.venv`, используйте правильный скрипт активации:

```powershell
.\.venv\Scripts\Activate.ps1
```

Если PowerShell запрещает запуск скриптов, временно разрешите их для текущего процесса:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

После этого снова запустите:

```powershell
.\.venv\Scripts\Activate.ps1
```

Альтернатива через `cmd.exe`:

```cmd
.venv\Scripts\activate.bat
```

### Windows: если `Activate.ps1` не найден и `pip` не запускается

Это означает, что виртуальное окружение **не было создано** или Python установлен некорректно.

1) Проверьте, что Python установлен и доступен:

```powershell
py -3.11 --version
```

Если команда не найдена — установите Python 3.11+ с https://www.python.org/downloads/ и отметьте галочку **“Add Python to PATH”**.

2) Создайте окружение через launcher `py`:

```powershell
py -3.11 -m venv .venv
```

3) Убедитесь, что скрипты появились:

```powershell
Get-ChildItem .\.venv\Scripts
```

4) Запускайте pip через интерпретатор окружения (работает даже без активации):

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install opencv-python numpy Pillow PySide6
```

## Запуск

```bash
python main.py
```

## Формат входных файлов

Файлы должны иметь имя в формате:

- `x,y.jpg` / `x,y.jpeg`
- `x,y_anySuffix.jpg` / `x,y_anySuffix.jpeg`

где:

- `x` — номер строки сверху вниз (нумерация с 1),
- `y` — номер в строке слева направо (нумерация с 1).

Пример:

- `1,1.jpg` → строка 1, колонка 1
- `1,2_scan.jpg` → строка 1, колонка 2
- `2,1.jpg` → строка 2, колонка 1

Файлы, не подходящие под формат, игнорируются.

## Как пользоваться GUI

1. В блоке **"Входные файлы"** выберите:
   - **"Выбрать файлы..."** (multi-select), или
   - **"Выбрать папку..."** (берутся все JPG).
2. В блоке **"Путь сохранения"** укажите итоговый `.tif`.
3. В **"Параметрах"** настройте:
   - `bg-threshold` (200–255, по умолчанию 245)
   - `max-angle` (0–20, по умолчанию 7.0)
   - `overlap-max` (0–30, по умолчанию 15)
   - `compression` (deflate/lzw/none)
   - `Debug режим` (сохраняет debug/ рядом с output)
4. Нажмите **"Старт"** и дождитесь завершения.
5. В правой панели появится **превью раскладки**.

## Как работает алгоритм

### 1) Парсинг координат

`parse_tile_coords` извлекает `(x, y)` из имени файла по шаблону `\d+,\d+(?:_.*)?\.(jpg|jpeg)`.

### 2) Предобработка каждого тайла

1. **Чтение в RGB**.
2. **Маска контента**: белые пиксели (все каналы > `bg-threshold`) исключаются.
3. **Обрезка** белых полей по bounding box маски.
4. **Оценка угла** по контуру маски + `minAreaRect`, ограничение `|angle| <= max-angle`.
5. **Поворот** и повторная обрезка.

### 3) Черновая раскладка

Тайлы группируются по строкам `x`, сортируются по `y` и раскладываются «вплотную».

### 4) Точная стыковка + авто-перекрытие

Для каждого тайла (кроме первого в строке/колонке):

- сравниваются полосы соседей (левый/верхний),
- ищется лучший сдвиг `(dx, dy)` в пределах `[-overlap-max, +overlap-max]`,
- используется NCC по градиентам (Sobel).

При слабом совпадении fallback: стык по краю без перекрытия.

### 5) Компоновка

- вычисляется общий холст,
- тайлы вклеиваются,
- при перекрытиях используется узкий **alpha-blend** только по зоне overlap.

### 6) Сохранение TIFF

- `compression`: `deflate` (default), `lzw` или `none`,
- BigTIFF активируется автоматически при больших размерах.

## Структура проекта

```
.
├── main.py
├── models.py
├── stitcher.py
├── utils.py
├── worker.py
└── README.md
```

## Отладка

Флаг **Debug режим** сохраняет папку `debug/` рядом с output:

- `*_crop_before.png`, `*_crop_after.png`
- JSON с результатами matching
- `positions.json` с финальными координатами
