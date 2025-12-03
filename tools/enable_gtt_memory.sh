#!/bin/bash
# Enable GTT (system RAM) for AMD iGPU with ROCm
# Based on: https://github.com/ollama/ollama/pull/6282

echo "=== Enabling GTT Memory for AMD Radeon 780M ==="
echo ""

# 1. Configure TTM (Translation Table Manager) for more GTT memory
echo "Configuring TTM kernel module..."

# Calculate pages for ~48GB GTT (pages_limit * 4KB = memory)
# 12582912 pages * 4KB = 48GB
TTM_PAGES=12582912

# Create modprobe config
sudo tee /etc/modprobe.d/ttm.conf > /dev/null <<EOF
# Enable more GTT memory for AMD iGPU (Radeon 780M)
# Each page = 4KB, so 12582912 * 4KB = 48GB
options ttm pages_limit=${TTM_PAGES}
options ttm page_pool_size=${TTM_PAGES}
EOF

echo "✓ Created /etc/modprobe.d/ttm.conf"
echo "  Pages limit: $TTM_PAGES (48GB)"

# 2. Update initramfs to apply changes
echo ""
echo "Updating initramfs..."
sudo update-initramfs -u

echo ""
echo "✓ GTT memory configuration complete!"
echo ""
echo "IMPORTANT: Reboot required for changes to take effect:"
echo "  sudo reboot"
echo ""
echo "After reboot, verify with:"
echo "  rocm-smi --showmeminfo all | grep GTT"
echo ""
echo "Expected: GTT Total Memory ~48GB"
