# Doorman Unified Benchmark System

Конфигурируемая система бенчмарков для тестирования производительности Doorman.

## Использование

### Быстрый запуск (без конфига)

```bash
# Просто запустить с параметрами по умолчанию
python3 tools/benchmark.py

# Указать режим и backend
python3 tools/benchmark.py --mode detection --backend torch-direct --iterations 100

# Доступные режимы:
#   detection       - только детекция лиц
#   liveness        - только проверка живости
#   embedding       - только извлечение эмбеддингов
#   full_pipeline   - полный пайплайн (detection + liveness + embedding)

# Доступные backends:
#   torch-direct    - прямой вызов Python (без IPC)
#   torch-ipc       - через Rust IPC (TODO)
#   tract           - Tract backend (TODO)
#   ort             - ONNX Runtime backend (TODO)
```

### С файлом конфигурации

```bash
# Один бенчмарк
python3 tools/benchmark.py -c tools/benchmark_configs/torch_detection.json

# Несколько бенчмарков последовательно
python3 tools/benchmark.py -c tools/benchmark_configs/all_modes.json

# Указать папку для результатов
python3 tools/benchmark.py -c config.json -o my_results/
```

## Формат конфигурации

### Один бенчмарк

```json
{
  "name": "PyTorch Detection (Direct)",
  "mode": "detection",
  "backend": "torch-direct",
  "iterations": 100,
  "warmup_iterations": 10,
  "image_width": 1024,
  "image_height": 720,
  "device": "cuda"
}
```

### Несколько бенчмарков

```json
[
  {
    "name": "Test 1",
    "mode": "detection",
    ...
  },
  {
    "name": "Test 2",
    "mode": "full_pipeline",
    ...
  }
]
```

## Параметры конфигурации

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `name` | string | да | Название бенчмарка |
| `mode` | string | да | Режим: `detection`, `liveness`, `embedding`, `full_pipeline` |
| `backend` | string | да | Backend: `torch-direct`, `torch-ipc`, `tract`, `ort` |
| `iterations` | int | нет | Количество итераций (по умолчанию: 100) |
| `warmup_iterations` | int | нет | Warmup итераций (по умолчанию: 10) |
| `image_width` | int | нет | Ширина тестового изображения (по умолчанию: 1024) |
| `image_height` | int | нет | Высота тестового изображения (по умолчанию: 720) |
| `device` | string | нет | Устройство: `cuda` или `cpu` (по умолчанию: `cuda`) |
| `models_dir` | string | нет | Путь к моделям (по умолчанию: `~/.local/share/doorman/models`) |

## Результаты

Результаты сохраняются в папку `benchmark_results/` (или указанную через `-o`) в формате JSON:

```
benchmark_results/
  torch-direct_detection_20251203_144530.json
  torch-direct_full_pipeline_20251203_144545.json
```

### Формат результата

```json
{
  "config": { ... },
  "mean_time_ms": 16.66,
  "std_time_ms": 1.54,
  "mean_fps": 60.03,
  "median_fps": 62.06,
  "first_10_avg_ms": 16.76,
  "last_10_avg_ms": 16.73,
  "degradation_percent": -0.2,
  "stable": true,
  "times": [...]
}
```

## Примеры

### Сравнить iGPU vs CPU

```bash
python3 tools/benchmark.py -c tools/benchmark_configs/all_modes.json
```

### Быстрый тест

```bash
python3 tools/benchmark.py --iterations 50 --warmup 5
```

### Тест стабильности (долгий)

```bash
python3 tools/benchmark.py --iterations 1000
```

## Интерпретация результатов

### Throughput (FPS)
- **Mean FPS**: средняя производительность
- **Median FPS**: медианная производительность (более устойчива к выбросам)
- **Min/Max FPS**: диапазон производительности

### Performance Stability
- **Degradation**: изменение производительности за время теста
  - `< -5%`: улучшение (JIT прогрелся)
  - `-5% to +5%`: стабильно ✓
  - `+5% to +20%`: небольшая деградация ⚠️
  - `> +20%`: серьёзная проблема ❌

### Типичные значения

| Backend | Mode | Expected FPS (iGPU) |
|---------|------|---------------------|
| torch-direct | detection | ~60 FPS |
| torch-direct | full_pipeline | ~30-40 FPS |
| torch-ipc | detection | ~8 FPS (с деградацией) |

## TODO

- [ ] Реализовать `torch-ipc` backend (через Rust)
- [ ] Реализовать `tract` backend
- [ ] Реализовать `ort` backend
- [ ] Добавить режим `liveness` и `embedding`
- [ ] Добавить сравнительные графики
- [ ] Добавить профилирование памяти
