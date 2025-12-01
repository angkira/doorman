#!/usr/bin/env python3
"""
AMD Ryzen AI NPU Diagnostic Script
Verifies NPU device access, environment, and library loading
"""

import os
import sys
import ctypes
from pathlib import Path

# ANSI color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def print_pass(text):
    print(f"{Colors.GREEN}✓ PASS:{Colors.RESET} {text}")

def print_fail(text):
    print(f"{Colors.RED}✗ FAIL:{Colors.RESET} {text}")

def print_warn(text):
    print(f"{Colors.YELLOW}⚠ WARN:{Colors.RESET} {text}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ INFO:{Colors.RESET} {text}")

def check_device_access():
    """Check if NPU device node exists and is accessible"""
    print_header("1. NPU Device Access Check")

    device_path = "/dev/accel/accel0"
    results = {"exists": False, "readable": False, "writable": False}

    # Check existence
    if os.path.exists(device_path):
        print_pass(f"Device node exists: {device_path}")
        results["exists"] = True

        # Check permissions
        if os.access(device_path, os.R_OK):
            print_pass(f"Device is readable by current user ({os.getlogin()})")
            results["readable"] = True
        else:
            print_fail(f"Device is NOT readable by current user")
            print_info(f"Run: sudo chmod 666 {device_path}")

        if os.access(device_path, os.W_OK):
            print_pass(f"Device is writable by current user")
            results["writable"] = True
        else:
            print_fail(f"Device is NOT writable by current user")
            print_info(f"Run: sudo chmod 666 {device_path}")

        # Show device info
        try:
            stat_info = os.stat(device_path)
            print_info(f"Device permissions: {oct(stat_info.st_mode)[-3:]}")
            print_info(f"Device owner: UID={stat_info.st_uid}, GID={stat_info.st_gid}")
        except Exception as e:
            print_warn(f"Could not stat device: {e}")
    else:
        print_fail(f"Device node does not exist: {device_path}")
        print_info("Check if amdxdna kernel module is loaded:")
        print_info("  lsmod | grep amdxdna")
        print_info("  dmesg | grep -i amdxdna")

    return all(results.values())

def check_environment():
    """Check required environment variables"""
    print_header("2. Environment Variables Check")

    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    required_paths = [
        '/opt/xilinx/xrt/lib',
        '/usr/local/lib'
    ]

    print_info(f"Current LD_LIBRARY_PATH: {ld_library_path or '(not set)'}")

    all_present = True
    for req_path in required_paths:
        if req_path in ld_library_path:
            print_pass(f"Found in LD_LIBRARY_PATH: {req_path}")
        else:
            print_fail(f"Missing from LD_LIBRARY_PATH: {req_path}")
            all_present = False

    if not all_present:
        print_warn("LD_LIBRARY_PATH is incomplete!")
        export_cmd = f"export LD_LIBRARY_PATH={':'.join(required_paths)}:$LD_LIBRARY_PATH"
        print_info(f"Add to your ~/.bashrc or run temporarily:")
        print(f"\n    {Colors.YELLOW}{export_cmd}{Colors.RESET}\n")

    # Check XILINX_XRT
    xilinx_xrt = os.environ.get('XILINX_XRT')
    if xilinx_xrt:
        print_pass(f"XILINX_XRT is set: {xilinx_xrt}")
    else:
        print_warn("XILINX_XRT is not set (optional but recommended)")
        print_info("Run: export XILINX_XRT=/opt/xilinx/xrt")

    return all_present

def check_library_loading():
    """Attempt to load XRT and driver libraries"""
    print_header("3. Library Loading Check")

    libraries = [
        {
            "name": "XRT Core Library",
            "paths": [
                "/opt/xilinx/xrt/lib/libxrt_core.so",
                "/opt/xilinx/xrt/lib/libxrt_core.so.2",
            ]
        },
        {
            "name": "XRT CoreUtil Library",
            "paths": [
                "/opt/xilinx/xrt/lib/libxrt_coreutil.so",
                "/opt/xilinx/xrt/lib/libxrt_coreutil.so.2",
            ]
        },
        {
            "name": "AMD XDNA Driver Shim",
            "paths": [
                "/usr/local/lib/libamdxdna.so",
                "/usr/local/lib/libxrt_plugin.so",
                "/usr/local/lib/libamdxdna_plugin.so",
            ]
        },
    ]

    loaded_count = 0
    total_count = len(libraries)

    for lib_info in libraries:
        lib_name = lib_info["name"]
        lib_paths = lib_info["paths"]

        loaded = False
        found_path = None

        # Try each possible path
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                found_path = lib_path
                try:
                    lib = ctypes.CDLL(lib_path)
                    print_pass(f"{lib_name}: Loaded successfully")
                    print_info(f"  Path: {lib_path}")
                    loaded = True
                    loaded_count += 1
                    break
                except OSError as e:
                    print_fail(f"{lib_name}: Found but failed to load")
                    print_info(f"  Path: {lib_path}")
                    print_warn(f"  Error: {e}")
                    break

        if not loaded and not found_path:
            print_fail(f"{lib_name}: Not found in any expected location")
            print_info(f"  Searched paths:")
            for p in lib_paths:
                print(f"    - {p}")

    print(f"\n{Colors.BOLD}Summary: {loaded_count}/{total_count} libraries loaded successfully{Colors.RESET}")
    return loaded_count == total_count

def check_xrt_tools():
    """Check if XRT command-line tools are accessible"""
    print_header("4. XRT Tools Check")

    xrt_bin_path = "/opt/xilinx/xrt/bin"
    tools = ["xrt-smi", "xbutil"]

    if not os.path.exists(xrt_bin_path):
        print_fail(f"XRT bin directory not found: {xrt_bin_path}")
        return False

    all_found = True
    for tool in tools:
        tool_path = os.path.join(xrt_bin_path, tool)
        if os.path.exists(tool_path) and os.access(tool_path, os.X_OK):
            print_pass(f"Found executable: {tool}")
            print_info(f"  Path: {tool_path}")
        else:
            print_fail(f"Not found or not executable: {tool}")
            all_found = False

    if all_found:
        print_info("You can run XRT tools with:")
        print(f"  {xrt_bin_path}/xrt-smi examine")
        print(f"  {xrt_bin_path}/xbutil examine")

    return all_found

def check_kernel_module():
    """Check if amdxdna kernel module is loaded"""
    print_header("5. Kernel Module Check")

    try:
        with open("/proc/modules", "r") as f:
            modules = f.read()

        if "amdxdna" in modules:
            print_pass("amdxdna kernel module is loaded")
            # Try to get module info
            try:
                import subprocess
                result = subprocess.run(
                    ["modinfo", "amdxdna"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if line.startswith('version:') or line.startswith('filename:'):
                            print_info(f"  {line}")
            except:
                pass
            return True
        else:
            print_fail("amdxdna kernel module is NOT loaded")
            print_info("Load it with: sudo modprobe amdxdna")
            return False
    except Exception as e:
        print_warn(f"Could not check kernel modules: {e}")
        return False

def main():
    print(f"\n{Colors.BOLD}AMD Ryzen AI NPU Diagnostic Script{Colors.RESET}")
    print(f"System: {os.uname().sysname} {os.uname().release}")
    print(f"User: {os.getlogin()} (UID: {os.getuid()})")

    results = {
        "device": check_device_access(),
        "environment": check_environment(),
        "libraries": check_library_loading(),
        "tools": check_xrt_tools(),
        "kernel": check_kernel_module(),
    }

    # Final summary
    print_header("Final Summary")

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for check_name, passed_check in results.items():
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed_check else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {check_name.capitalize():20s}: {status}")

    print(f"\n{Colors.BOLD}Overall: {passed}/{total} checks passed{Colors.RESET}")

    if all(results.values()):
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ NPU environment appears to be configured correctly!{Colors.RESET}")
        print_info("You can now try using ONNX Runtime with VitisAI execution provider")
        return 0
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ Some checks failed. Review the output above for fixes.{Colors.RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
