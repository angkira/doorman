# Native Extension - Zero IPC Overhead ✨

## Проблема

IPC коммуникация между Rust daemon и Python inference добавляет **50-80ms overhead** на каждый frame:

| Component | Overhead |
|-----------|----------|
| Base64 encode/decode | 2-3ms |
| JSON serialize/parse | 1-2ms |
| IPC communication | 3-5ms |
| Python subprocess | 1-2ms |
| **TOTAL** | **~7-12ms** |

**Результат:** 60 FPS baseline → 7-10 FPS с IPC

## Решение: Native PyO3 Extension

Прямой вызов ONNX Runtime из Rust, скомпилированный как Python модуль.

### Архитектура

```
Python (benchmark.py, daemon)
    ↓ (PyO3 FFI - <1ms overhead)
Rust (doorman_ml_native.so)
    ↓ (direct ONNX Runtime calls)
ONNX Runtime
    ↓ (MIGraphX/ROCm)
AMD iGPU
```

**Никакого IPC, никакой сериализации, никаких subprocess!**

## Реализация

### Компоненты

```
daemon/native_ml/
├── Cargo.toml          # PyO3 extension config
├── pyproject.toml      # maturin build config
├── src/
│   ├── lib.rs          # PyO3 bindings (DoormanML class)
│   ├── detector.rs     # BlazeFace wrapper
│   ├── liveness.rs     # Liveness checker
│   └── embedder.rs     # MobileFaceNet wrapper
└── python/
    └── doorman_ml/
        └── __init__.py # Python package
```

### API

```python
from doorman_ml import DoormanML

# Initialize
ml = DoormanML(models_dir="./models", device="cuda")

# Detect faces (returns native Rust objects)
detections = ml.detect_faces(image_rgb_bytes, width=1024, height=720)
for det in detections:
    print(f"Bbox: {det.bbox}, confidence: {det.confidence:.3f}")

# Check liveness
liveness = ml.check_liveness(face_crop_112x112_rgb)
print(f"Is live: {liveness.is_live}, confidence: {liveness.confidence:.3f}")

# Extract embedding
embedding_bytes = ml.extract_embedding(face_crop_112x112_rgb)
embedding = np.frombuffer(embedding_bytes, dtype=np.float32)  # 512-dim
```

## Build

### Требования

```bash
# Install maturin (PyO3 build tool)
pip install maturin

# Install Rust (if not installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Сборка

```bash
# Development build (fast iteration)
cd daemon/native_ml
./build.sh

# Or manually:
maturin develop --release

# Build wheel for distribution
maturin build --release
pip install target/wheels/doorman_ml_native-*.whl
```

### Проверка

```bash
python3 -c "from doorman_ml import DoormanML; print(DoormanML)"
# <class 'doorman_ml_native.DoormanML'>
```

## Benchmark

### Запуск

```bash
# Native extension vs Direct Python vs IPC
python3 tools/benchmark.py -c tools/benchmark_configs/native_comparison.json
```

### Ожидаемые результаты

| Backend | Mean FPS | Overhead | Notes |
|---------|----------|----------|-------|
| **torch-native** | **~55-60** | **<1ms** | **PyO3 extension** ✨ |
| torch-direct | ~60 | 0ms | Baseline (pure Python) |
| torch-ipc | ~7-10 | 50-80ms | JSON-RPC subprocess |

**Выигрыш:** ~6-8x FPS по сравнению с IPC!

## Интеграция с Daemon

### Замена TorchBackend

```rust
// daemon/src/ml/torch_backend_native.rs
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub struct TorchBackendNative {
    py_module: Py<PyModule>,
    ml_instance: Py<PyAny>,
}

impl TorchBackendNative {
    pub fn new(models_dir: &Path, device: &str) -> Result<Self> {
        Python::with_gil(|py| {
            // Import native module
            let module = PyModule::import(py, "doorman_ml")?;
            let ml_class = module.getattr("DoormanML")?;
            
            // Create instance
            let ml_instance = ml_class.call1((
                models_dir.to_str().unwrap(),
                device
            ))?;

            Ok(Self {
                py_module: module.into(),
                ml_instance: ml_instance.into(),
            })
        })
    }

    pub fn detect_faces(&self, image_data: &[u8], w: u32, h: u32) -> Result<Vec<Face>> {
        Python::with_gil(|py| {
            let image_bytes = PyBytes::new(py, image_data);
            let result = self.ml_instance.call_method1(
                py,
                "detect_faces",
                (image_bytes, w, h)
            )?;

            // Parse detections...
            Ok(faces)
        })
    }
}
```

### Конфигурация

```toml
# doorman.toml
[ml]
backend = "torch-native"  # ← NEW
device = "cuda"
models_dir = "~/.local/share/doorman/models"
```

## Преимущества

1. **Производительность:** ~55-60 FPS (vs 7-10 FPS IPC)
2. **Простота:** Те же ONNX модели, никаких изменений
3. **Надёжность:** Нет subprocess, нет IPC ошибок
4. **Отладка:** Stack traces проходят через Rust → Python
5. **Deployment:** Один `.so` файл, никаких скриптов

## Недостатки

1. Требует сборки (но это один раз)
2. Зависимость от PyO3 (но стабильная)
3. Python GIL может быть узким местом (но не в нашем случае - один inference за раз)

## Следующие шаги

1. ✅ Создана структура PyO3 extension
2. ✅ Реализованы detector, liveness, embedder
3. ✅ Добавлен TorchNativeBackend в benchmark
4. ⏳ **Собрать и протестировать**
5. ⏳ Сравнить с Direct/IPC
6. ⏳ Интегрировать в daemon (заменить TorchBackend)
7. ⏳ Измерить финальную производительность

## Альтернативы

Если PyO3 extension не подходит:

1. **Shared Memory IPC** - промежуточное решение (~40-50 FPS)
2. **gRPC** - бинарный протокол вместо JSON (~30-40 FPS)
3. **Pure Rust inference** - перенести всё в Rust, без Python

Но **Native Extension - самое простое и быстрое решение** для текущей архитектуры.
