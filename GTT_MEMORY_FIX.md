## GTT Memory Fix для AMD Radeon 780M

### Проблема

ONNX Runtime MIGraphX использует только **2GB VRAM**, игнорируя **32GB GTT** (system RAM).

```bash
rocm-smi --showmeminfo all
# VRAM Total: 2GB (89% used) ❌
# GTT Total: 32GB (available!)  ✓
```

### Решение

**Option 1: Kernel TTM Configuration (рекомендуется)**

Настроить TTM (Translation Table Manager) для большего GTT:

```bash
./tools/enable_gtt_memory.sh
sudo reboot
```

После reboot проверить:
```bash
rocm-smi --showmeminfo all | grep GTT
# Должно показать ~48GB GTT
```

**Option 2: Environment Variables (без reboot)**

Обновлённый `tools/run_torch.sh` уже включает:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.1  # было 11.0.0
export HIP_VISIBLE_DEVICES=0
export GPU_MAX_HW_QUEUES=1
```

Запустить:
```bash
./tools/run_torch.sh --preview
```

### Почему это работает

**Linux Kernel 6.10+** изменил аллокацию памяти:
- **До 6.10**: ROCm использовал только VRAM
- **После 6.10**: ROCm может использовать GTT (system RAM)

У тебя **kernel 6.17** ✅ - поддерживается!

**TTM settings:**
- `pages_limit`: максимум GTT страниц
- `page_pool_size`: размер пула страниц
- Каждая страница = 4KB
- 12582912 pages × 4KB = **48GB GTT**

### Источники

- [Ollama AMD iGPU GTT fix](https://github.com/ollama/ollama/pull/6282)
- [Available memory calculation on AMD APU](https://github.com/ollama/ollama/issues/5471)
- [Running Ollama on AMD iGPU](https://blog.machinezoo.com/Running_Ollama_on_AMD_iGPU)
- [ROCm Radeon Prerequisites](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/prerequisites.html)

### Alternative: pytorch-rocm-gtt

Для PyTorch есть патч:

```bash
pip install pytorch-rocm-gtt
```

Но для ONNX Runtime нужны TTM настройки.

### Verification

После применения fix:

```bash
# Check GTT
rocm-smi --showmeminfo all

# Run daemon
./tools/run_torch.sh --preview

# Should see in logs:
# [PyTorch] Model on device: MIGraphXExecutionProvider
# Detection FPS: 100-120 (not 0.5!)
```

### Troubleshooting

**Если всё ещё OOM:**

1. Закрыть тяжёлые desktop приложения
2. Убить plasmashell временно:
   ```bash
   killall plasmashell
   # Тестировать
   # Восстановить: kstart plasmashell
   ```

3. Использовать FP16:
   ```python
   # В torch_inference.py
   provider_options = {
       'migraphx_fp16_enable': '1',  # Half precision
   }
   ```

### Status

- ✅ Kernel 6.17 (поддерживает GTT)
- ✅ ROCm 7.0.2
- ✅ 32GB GTT доступно
- ⏳ TTM конфигурация (нужен reboot)
- ⏳ Тест с новыми env vars
