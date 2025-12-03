# GPU Memory Issue - iGPU с Desktop Compositor

## Проблема

**HIP failure 719: unspecified launch failure**

```
VRAM Total: 2GB
VRAM Used: 1.9GB (Desktop compositor)
Available: ~230MB

Требуется для inference:
- BlazeFace: ~50-100MB
- Liveness: ~50-100MB
- MobileFaceNet: ~250-300MB
Итого: ~400-500MB ❌
```

## Причина

iGPU (Radeon 780M) делит память с системой и используется для:
1. **Desktop rendering** (KWin Wayland + Plasmashell) - **1.9GB**
2. **ML inference** (MIGraphX) - нужно **400-500MB**

**Конфликт:** Desktop уже занял почти всю доступную VRAM.

## GPU Configuration

Система имеет 2 GPU:
1. **NVIDIA RTX 4060 Ti** (card0) - dGPU, НЕ поддерживается ROCm ⚠️
2. **AMD Radeon 780M** (card1) - iGPU, shared VRAM 2GB

ROCm работает ТОЛЬКО с AMD GPU → можем использовать только iGPU.

## Решения

### Option 1: CPU Backend ✅ (рекомендуется)

**Стабильно работает**, без GPU конфликтов:

```bash
./tools/run_torch_cpu.sh --preview
```

**Производительность:**
- Detection: ~10-20 FPS (vs 100-120 на GPU)
- Full pipeline: ~5-10 FPS (vs 20-50 на GPU)
- Стабильность: ✅ 100%

### Option 2: Закрыть Desktop для освобождения VRAM

**Не практично** - нужен GUI.

### Option 3: NVIDIA CUDA Backend (будущая задача)

Для использования RTX 4060 Ti нужно:

1. Установить `onnxruntime-gpu` вместо `onnxruntime-rocm`:
   ```bash
   uv pip uninstall onnxruntime-rocm
   uv pip install onnxruntime-gpu
   ```

2. Изменить providers в `torch_inference.py`:
   ```python
   providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
   # или
   providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
   ```

3. Настроить CUDA environment:
   ```bash
   export CUDA_VISIBLE_DEVICES=0  # NVIDIA GPU
   ```

**Преимущества:**
- RTX 4060 Ti имеет **8GB VRAM** (vs 2GB iGPU)
- Dedicated GPU (не конфликтует с desktop)
- TensorRT очень быстрый

**Недостатки:**
- Нужна установка CUDA toolkit
- Нужно переключить backend
- Отдельная задача

### Option 4: Hybrid (Desktop на iGPU, Inference на CPU)

Текущее решение - использовать CPU для inference.

## Workaround для iGPU (экспериментально)

Если очень хочется использовать iGPU:

### 1. Уменьшить VRAM desktop'а

Закрыть тяжёлые приложения, использующие GPU:

```bash
# Проверить что использует GPU
lsof /dev/dri/renderD128

# Убить plasmashell (освободит ~500MB)
killall plasmashell
# Перезапустить позже: kstart plasmashell
```

### 2. Использовать FP16 (half precision)

В `torch_inference.py` добавить:

```python
provider_options = {
    'device_id': 0,
    'migraphx_fp16_enable': '1',  # Use FP16 (2x меньше VRAM)
}
```

**Риски:**
- Меньшая точность
- Возможны ошибки в результатах

### 3. Offload некоторых моделей на CPU

Использовать GPU только для BlazeFace (самая тяжёлая по FPS), остальное на CPU.

## Рекомендации

### Для production:

1. **Сейчас:** Использовать CPU backend
   ```bash
   ./tools/run_torch_cpu.sh --preview
   ```

2. **Лучше:** Настроить NVIDIA CUDA backend
   - 8GB VRAM
   - Dedicated GPU
   - Нет конфликтов

### Для headless серверов:

Если doorman запускается на сервере без desktop:
```bash
./tools/run_torch.sh --preview  # iGPU будет свободен
```

## Performance Comparison

| Backend | Device | VRAM | Detection FPS | Status |
|---------|--------|------|---------------|--------|
| MIGraphX | iGPU (Radeon 780M) | 2GB shared | 100-120 | ❌ Out of memory |
| CPU | CPU (Ryzen 7 8700G) | N/A | 10-20 | ✅ Stable |
| TensorRT | dGPU (RTX 4060 Ti) | 8GB dedicated | 200-300+ | ⏳ Not configured |

## Files

**CPU Backend:**
- `tools/run_torch_cpu.sh` - Launch script (CPU mode)
- `tools/configs/doorman-torch-cpu.toml` - Config with `device = "cpu"`

**GPU Backend (broken):**
- `tools/run_torch.sh` - Launch script (GPU mode) ⚠️ OOM
- `tools/configs/doorman-torch.toml` - Config with `device = "cuda"`

## Next Steps

1. **Short term:** Use CPU backend ✅
2. **Long term:** Configure NVIDIA CUDA backend for RTX 4060 Ti
3. **Alternative:** Use headless mode (no desktop) to free iGPU VRAM

## Debug Commands

```bash
# Check VRAM usage
rocm-smi --showmeminfo vram

# Check GPU processes
lsof /dev/dri/renderD128

# Check desktop memory
ps aux | grep -E "kwin|plasma" | awk '{sum+=$6} END {print sum/1024 " MB"}'

# Free VRAM (kill desktop)
killall plasmashell kwin_wayland
```
