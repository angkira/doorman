# MIGraphX Model Caching

## TL;DR

**Да, можно предкомпилировать!** MIGraphX автоматически кэширует compiled модели.

## Как работает кэширование

### Автоматический кэш (ROCm 6.4+)

MIGraphX использует environment variables для управления кэшем:

```bash
export ORT_MIGRAPHX_MODEL_CACHE_PATH="$HOME/.cache/doorman/migraphx"
export ORT_MIGRAPHX_MODEL_PATH="$HOME/.cache/doorman/migraphx"
```

**Первый запуск:**
1. Модель загружается из `.onnx` файла
2. MIGraphX компилирует для GPU (долго)
3. Сохраняет скомпилированный `.mxr` в кэш

**Последующие запуски:**
1. MIGraphX проверяет кэш
2. Находит скомпилированный `.mxr`
3. **Загружает мгновенно** (без компиляции)

### Время компиляции vs загрузки

| Модель | Размер | Первый запуск | Из кэша |
|--------|--------|---------------|---------|
| BlazeFace | 1.3MB | ~30s | ~1s |
| Liveness | 1.3MB | ~6s | <1s |
| MobileFaceNet | 249MB | **3-5 мин** | ~2-3s |

## Предкомпиляция моделей

### Option 1: Автоматическая (рекомендуется)

**Просто запусти daemon один раз:**

```bash
./tools/run_torch.sh --preview
# Дождись "Model Compile: Complete" для всех моделей
# Ctrl+C после компиляции
```

MIGraphX сохранит кэш в `~/.cache/doorman/migraphx/`

### Option 2: Скрипт предкомпиляции

```bash
source .venv/bin/activate
python3 tools/precompile_models.py
```

Этот скрипт:
1. Загружает все 3 модели
2. Принудительно компилирует для GPU
3. Сохраняет в кэш
4. Следующий запуск daemon будет быстрым

## Проверка кэша

### Посмотреть скомпилированные модели:

```bash
ls -lh ~/.cache/doorman/migraphx/
```

Должны появиться `.mxr` файлы после первой компиляции.

### Очистить кэш (для переcompile):

```bash
rm -rf ~/.cache/doorman/migraphx/
```

## Технические детали

### Устаревшие опции (до ROCm 6.4)

**Не используй** эти session options - они удалены:
- `migraphx_save_compiled_model` ❌
- `migraphx_save_compiled_path` ❌
- `migraphx_load_compiled_model` ❌
- `migraphx_load_compiled_path` ❌

### Текущие опции (ROCm 6.4+)

**Используй** environment variables:
- `ORT_MIGRAPHX_MODEL_CACHE_PATH` ✅
- `ORT_MIGRAPHX_MODEL_PATH` ✅

## Integration в проект

### tools/run_torch.sh

Уже настроен:

```bash
export ORT_MIGRAPHX_MODEL_CACHE_PATH="$HOME/.cache/doorman/migraphx"
export ORT_MIGRAPHX_MODEL_PATH="$HOME/.cache/doorman/migraphx"
mkdir -p "$ORT_MIGRAPHX_MODEL_CACHE_PATH"
```

### daemon/src/ml/torch_inference.py

Читает env vars автоматически - ничего менять не нужно.

## Workflow для установки

### Первая установка (one-time setup):

```bash
# 1. Install dependencies
uv sync

# 2. Build daemon
cargo build --release --features backend-torch

# 3. Pre-compile models (optional but recommended)
source .venv/bin/activate
python3 tools/precompile_models.py

# 4. Daemon готов к использованию (fast startup)
./tools/run_torch.sh --preview
```

### Обычное использование (после setup):

```bash
./tools/run_torch.sh --preview
# Быстрый старт (модели из кэша)
```

## Troubleshooting

### Кэш не работает

Проверь env vars:

```bash
echo $ORT_MIGRAPHX_MODEL_CACHE_PATH
# Должно: /home/user/.cache/doorman/migraphx
```

### Модели перекомпилируются каждый раз

Возможные причины:
1. Кэш директория не создана → `mkdir -p ~/.cache/doorman/migraphx`
2. Нет прав на запись → `chmod 755 ~/.cache/doorman/migraphx`
3. ROCm версия < 6.4 → обнови ROCm

### Проверка версии ROCm:

```bash
rocminfo | grep "Runtime Version"
# Рекомендуется: ROCm 6.4+
```

## Performance Impact

### До кэширования:
- Первый запуск: **3-5 минут** (компиляция MobileFaceNet)
- Detection FPS: падает до 0.4 во время компиляции

### После кэширования:
- Запуск: **2-3 секунды** (загрузка из кэша)
- Detection FPS: **100-120** сразу
- Full pipeline FPS: **20-50**

## Рекомендации

1. **Всегда используй кэш** - добавь env vars в run script ✅
2. **Pre-compile при установке** - запусти `precompile_models.py` один раз
3. **Не удаляй кэш** - он ускоряет каждый запуск
4. **Backup кэша** - можно скопировать `.cache/doorman/migraphx/` на другие машины с тем же GPU

## Sources

- [ONNX Runtime MIGraphX EP](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)
- [MIGraphX Environment Variables](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/dev/env_vars.html)
- [EP Context Design](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html)
