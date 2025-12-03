# Unified Benchmark System - Готово! ✓

## Что сделано

1. **Удалены сломанные бенчмарки:**
   - ❌ `daemon/benches/full_pipeline.rs` - не компилируется (API изменился)
   - ❌ `tools/benchmark_python_backend.sh` - полагался на сломанный Rust бенчмарк
   - ❌ `tools/benchmark_detection.py` - временный скрипт

2. **Создана единая система бенчмарков:**
   - ✓ `tools/benchmark.py` - конфигурируемый бенчмарк
   - ✓ `tools/BENCHMARK_README.md` - полная документация
   - ✓ `tools/benchmark_configs/` - примеры конфигураций

## Использование

### Быстрый запуск

```bash
# Простой запуск с параметрами по умолчанию
python3 tools/benchmark.py

# С параметрами
python3 tools/benchmark.py --iterations 100 --warmup 10

# С конфигурацией
python3 tools/benchmark.py -c tools/benchmark_configs/torch_detection.json
```

### Доступные конфигурации

- `torch_detection.json` - PyTorch detection на iGPU
- `all_modes.json` - несколько бенчмарков: iGPU, CPU, full pipeline

## Результаты (AMD Radeon 780M iGPU)

### PyTorch Direct (без IPC overhead)

```json
{
  "backend": "torch-direct",
  "mode": "detection",
  "mean_fps": 61.46,
  "median_fps": 62.49,
  "degradation_percent": -2.19,
  "stable": true
}
```

**Выводы:**
- ✅ **60 FPS** стабильно на iGPU
- ✅ Производительность **стабильна** (деградация -2.2%, даже улучшение!)
- ✅ Python inference **работает отлично**

**Проблема в IPC слое:**
- Daemon через Rust IPC: 7.8 → 2.1 FPS (деградация)
- Прямой Python: 60 FPS (стабильно)
- **Узкое место: JSON-RPC коммуникация Rust ↔ Python**

## Архитектура

```
tools/benchmark.py
├── BenchmarkConfig (dataclass)
│   ├── name, mode, backend
│   ├── iterations, warmup_iterations
│   └── image_width, image_height, device
│
├── BenchmarkResult (dataclass)
│   ├── Statistics: mean, median, min, max
│   ├── FPS metrics
│   ├── Degradation tracking
│   └── to_dict() - JSON serialization
│
├── Backends:
│   ├── TorchDirectBackend (✓ реализовано)
│   ├── TorchIPCBackend (TODO)
│   ├── TractBackend (TODO)
│   └── ORTBackend (TODO)
│
└── BenchmarkRunner
    ├── Warmup
    ├── Iterations with progress
    ├── Statistics calculation
    └── JSON results
```

## Конфигурация

### Пример (один бенчмарк)

```json
{
  "name": "PyTorch Detection (iGPU)",
  "mode": "detection",
  "backend": "torch-direct",
  "iterations": 100,
  "warmup_iterations": 10,
  "image_width": 1024,
  "image_height": 720,
  "device": "cuda"
}
```

### Пример (несколько бенчмарков)

```json
[
  {
    "name": "PyTorch iGPU",
    "backend": "torch-direct",
    "mode": "detection",
    ...
  },
  {
    "name": "PyTorch CPU",
    "backend": "torch-direct",
    "mode": "detection",
    "device": "cpu",
    ...
  }
]
```

## Результаты сохраняются

```
benchmark_results/
  torch-direct_detection_20251203_145902.json
```

Формат:
```json
{
  "config": { ... },
  "mean_time_ms": 16.27,
  "mean_fps": 61.46,
  "degradation_percent": -2.19,
  "stable": true,
  "sample_count": 20
}
```

## TODO

- [ ] Реализовать `TorchIPCBackend` для измерения IPC overhead
- [ ] Реализовать `TractBackend` для сравнения
- [ ] Добавить режимы `liveness`, `embedding`, `full_pipeline`
- [ ] Добавить визуализацию результатов (графики)
- [ ] Добавить профилирование памяти
- [ ] Оптимизировать IPC слой (устранить деградацию)

## Следующие шаги

1. **Реализовать TorchIPCBackend** для точного измерения overhead
2. **Оптимизировать IPC:**
   - Уменьшить JSON-RPC overhead
   - Использовать shared memory для передачи изображений?
   - Батчинг запросов?
   - Пулл процессов вместо одного subprocess?
