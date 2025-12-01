import ctypes
import os
import sys

# Настройка путей (дублируем логику LD_LIBRARY_PATH внутри скрипта для надежности)
xrt_lib_path = "/opt/xilinx/xrt/lib"
shim_lib_path = "/usr/local/lib"

print(f"--- Diagnostic Start ---")
print(f"Checking access to Shim: {shim_lib_path}/libxrt_driver_xdna.so")

if not os.path.exists(f"{shim_lib_path}/libxrt_driver_xdna.so"):
    print("FATAL: Symlink libxrt_driver_xdna.so not found! Did you run Step 1?")
    sys.exit(1)

try:
    # 1. Сначала грузим XRT Core
    print("Loading XRT Core...")
    xrt_core = ctypes.CDLL(f"{xrt_lib_path}/libxrt_core.so", mode=ctypes.RTLD_GLOBAL)
    print("[PASS] XRT Core loaded.")

    # 2. Теперь ЯВНО грузим Shim драйвер.
    # Если тут упадет - значит несовместимость версий или зависимостей.
    print("Loading XDNA Driver Shim...")
    shim = ctypes.CDLL(
        f"{shim_lib_path}/libxrt_driver_xdna.so", mode=ctypes.RTLD_GLOBAL
    )
    print("[PASS] XDNA Driver Shim loaded directly via ctypes.")

except OSError as e:
    print(f"[FAIL] Library load error: {e}")
    sys.exit(1)

# 3. Вызываем xclProbe
print("\nCalling xclProbe to scan for devices...")
try:
    xrt_core.xclProbe.restype = ctypes.c_uint
    count = xrt_core.xclProbe()
    print(f"-----------------------------------")
    print(f"Devices found: {count}")
    print(f"-----------------------------------")

    if count > 0:
        print("SUCCESS! The NPU is visible.")
    else:
        print("Still 0 devices. Check dmesg for XDNA errors.")

except Exception as e:
    print(f"Error calling xclProbe: {e}")
