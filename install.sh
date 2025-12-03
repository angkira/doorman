#!/usr/bin/env bash
set -e

# Doorman Interactive Installer
# Configures and installs doorman daemon with appropriate ML backend

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Doorman Face Authentication System Installer          ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Detect system info
echo -e "${YELLOW}→${NC} Detecting system configuration..."
OS=$(uname -s)
ARCH=$(uname -m)
echo -e "  OS: ${GREEN}$OS${NC}"
echo -e "  Arch: ${GREEN}$ARCH${NC}"

# Detect GPU
GPU_TYPE="none"
if command -v rocm-smi &> /dev/null; then
    GPU_TYPE="amd"
    GPU_INFO=$(rocm-smi --showproductname 2>/dev/null | grep "GPU\[" | head -1 || echo "AMD GPU")
    echo -e "  GPU: ${GREEN}AMD ROCm${NC} ($GPU_INFO)"
elif command -v nvidia-smi &> /dev/null; then
    GPU_TYPE="nvidia"
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo -e "  GPU: ${GREEN}NVIDIA CUDA${NC} ($GPU_INFO)"
else
    echo -e "  GPU: ${YELLOW}None detected (CPU only)${NC}"
fi

echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check dependencies
echo -e "${YELLOW}→${NC} Checking dependencies..."

MISSING_DEPS=()

if ! command_exists rustc; then
    MISSING_DEPS+=("rustc (Rust compiler)")
fi

if ! command_exists cargo; then
    MISSING_DEPS+=("cargo (Rust package manager)")
fi

if ! command_exists python3; then
    MISSING_DEPS+=("python3")
fi

if ! command_exists pip3 && ! command_exists uv; then
    MISSING_DEPS+=("pip3 or uv (Python package manager)")
fi

if ! command_exists pkg-config; then
    MISSING_DEPS+=("pkg-config")
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${RED}✗${NC} Missing dependencies:"
    for dep in "${MISSING_DEPS[@]}"; do
        echo -e "  - $dep"
    done
    echo ""
    echo "Please install missing dependencies and run installer again."
    echo "Example (Arch Linux): sudo pacman -S rust python python-pip pkg-config"
    exit 1
fi

echo -e "${GREEN}✓${NC} All required dependencies found"
echo ""

# Select ML backend
echo -e "${YELLOW}→${NC} Select ML Backend:"
echo ""
echo "  1) ${GREEN}tract${NC}       - Pure Rust, CPU-only, no external deps"
echo "                    ${YELLOW}Performance: ~15-20 FPS (CPU)${NC}"
echo ""
echo "  2) ${GREEN}torch${NC}       - Python PyTorch + ONNX Runtime (recommended)"
echo "                    ${YELLOW}Performance: 50-60 FPS (GPU) / 8-10 FPS (CPU)${NC}"
echo "                    Requires: onnxruntime-rocm (AMD) or onnxruntime-gpu (NVIDIA)"
echo ""
echo "  3) ${GREEN}torch-native${NC} - Native PyO3 extension (experimental)"
echo "                    ${YELLOW}Performance: 169 FPS (theoretical)${NC}"
echo "                    ${RED}Warning: Complex setup, library path issues${NC}"
echo ""

read -p "Choose backend [1-3] (default: 2): " BACKEND_CHOICE
BACKEND_CHOICE=${BACKEND_CHOICE:-2}

case $BACKEND_CHOICE in
    1)
        BACKEND="tract"
        CARGO_FEATURES="backend-tract"
        PYTHON_DEPS=()
        ;;
    2)
        BACKEND="torch"
        CARGO_FEATURES="backend-torch"
        if [ "$GPU_TYPE" = "amd" ]; then
            PYTHON_DEPS=("onnxruntime-rocm")
        elif [ "$GPU_TYPE" = "nvidia" ]; then
            PYTHON_DEPS=("onnxruntime-gpu")
        else
            PYTHON_DEPS=("onnxruntime")
        fi
        ;;
    3)
        BACKEND="torch-native"
        CARGO_FEATURES="backend-torch-native"
        if [ "$GPU_TYPE" = "amd" ]; then
            PYTHON_DEPS=("onnxruntime-rocm")
        elif [ "$GPU_TYPE" = "nvidia" ]; then
            PYTHON_DEPS=("onnxruntime-gpu")
        else
            PYTHON_DEPS=("onnxruntime")
        fi
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}✓${NC} Selected: $BACKEND"
echo ""

# Select camera backend
echo -e "${YELLOW}→${NC} Select Camera Backend:"
echo ""
echo "  1) ${GREEN}gstreamer${NC} - GStreamer (recommended for desktop)"
echo "                    Supports PipeWire, V4L2, network streams"
echo ""
echo "  2) ${GREEN}v4l${NC}        - Video4Linux (simple, direct device access)"
echo "                    Works with /dev/videoN devices"
echo ""

read -p "Choose camera [1-2] (default: 1): " CAMERA_CHOICE
CAMERA_CHOICE=${CAMERA_CHOICE:-1}

case $CAMERA_CHOICE in
    1)
        CAMERA="gstreamer"
        CARGO_FEATURES="$CARGO_FEATURES,camera-gstreamer"
        ;;
    2)
        CAMERA="v4l"
        CARGO_FEATURES="$CARGO_FEATURES,camera-v4l"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}✓${NC} Selected: $CAMERA"
echo ""

# Installation directory
echo -e "${YELLOW}→${NC} Installation Mode:"
echo ""
echo "  1) ${GREEN}User${NC}   - Install for current user (~/.local)"
echo "                ~/.local/bin/doormand"
echo "                ~/.local/share/doorman/"
echo "                ~/.config/systemd/user/doormand.service"
echo ""
echo "  2) ${GREEN}System${NC} - Install system-wide (requires root)"
echo "                /usr/local/bin/doormand"
echo "                /var/lib/doorman/"
echo "                /etc/systemd/system/doormand.service"
echo ""

read -p "Choose mode [1-2] (default: 1): " INSTALL_MODE
INSTALL_MODE=${INSTALL_MODE:-1}

case $INSTALL_MODE in
    1)
        INSTALL_DIR="$HOME/.local"
        DATA_DIR="$HOME/.local/share/doorman"
        CONFIG_DIR="$HOME/.config/doorman"
        SERVICE_DIR="$HOME/.config/systemd/user"
        SOCKET_PATH="/run/user/$(id -u)/doorman.sock"
        SYSTEMD_TYPE="user"
        ;;
    2)
        if [ "$EUID" -ne 0 ]; then
            echo -e "${RED}✗${NC} System installation requires root. Run with sudo."
            exit 1
        fi
        INSTALL_DIR="/usr/local"
        DATA_DIR="/var/lib/doorman"
        CONFIG_DIR="/etc/doorman"
        SERVICE_DIR="/etc/systemd/system"
        SOCKET_PATH="/run/doorman.sock"
        SYSTEMD_TYPE="system"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}✓${NC} Install dir: $INSTALL_DIR"
echo ""

# Summary
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Installation Summary:${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "  Backend:       ${GREEN}$BACKEND${NC}"
echo -e "  Camera:        ${GREEN}$CAMERA${NC}"
echo -e "  Install mode:  ${GREEN}$SYSTEMD_TYPE${NC}"
echo -e "  Binary:        $INSTALL_DIR/bin/doormand"
echo -e "  Data:          $DATA_DIR"
echo -e "  Config:        $CONFIG_DIR"
echo -e "  Service:       $SERVICE_DIR/doormand.service"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

read -p "Proceed with installation? [y/N]: " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

echo ""
echo -e "${YELLOW}→${NC} Starting installation..."
echo ""

# Create directories
echo -e "${YELLOW}→${NC} Creating directories..."
mkdir -p "$INSTALL_DIR/bin"
mkdir -p "$DATA_DIR/models"
mkdir -p "$DATA_DIR/embeddings"
mkdir -p "$CONFIG_DIR"
mkdir -p "$SERVICE_DIR"
echo -e "${GREEN}✓${NC} Directories created"
echo ""

# Setup Python environment (for torch backends)
if [[ "$BACKEND" == "torch" || "$BACKEND" == "torch-native" ]]; then
    echo -e "${YELLOW}→${NC} Setting up Python environment..."
    
    # Create venv in data directory (persistent location)
    VENV_DIR="$DATA_DIR/venv"
    echo "  Creating venv at: $VENV_DIR"
    
    if command_exists uv; then
        echo "  Using uv for fast Python package management"
        uv venv "$VENV_DIR" --python python3.12 || uv venv "$VENV_DIR"
    else
        echo "  Using pip"
        python3.12 -m venv "$VENV_DIR" || python3 -m venv "$VENV_DIR"
    fi
    
    source "$VENV_DIR/bin/activate"
    
    # Install dependencies
    if command_exists uv; then
        for dep in "${PYTHON_DEPS[@]}"; do
            echo "  Installing $dep with uv..."
            uv pip install "$dep"
        done
    else
        pip install --upgrade pip
        for dep in "${PYTHON_DEPS[@]}"; do
            echo "  Installing $dep..."
            pip install "$dep"
        done
    fi
    
    # Get absolute path to onnxruntime lib and create symlink if needed
    ORT_CAPI_DIR=$(python3 -c "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))" 2>/dev/null || echo "")
    
    if [ -n "$ORT_CAPI_DIR" ] && [ -d "$ORT_CAPI_DIR" ]; then
        # Create symlink libonnxruntime.so -> libonnxruntime.so.X.Y.Z
        cd "$ORT_CAPI_DIR"
        if [ ! -e "libonnxruntime.so" ]; then
            VERSIONED_LIB=$(ls libonnxruntime.so.* 2>/dev/null | head -1)
            if [ -n "$VERSIONED_LIB" ]; then
                ln -sf "$VERSIONED_LIB" libonnxruntime.so
                echo "  Created symlink: libonnxruntime.so -> $VERSIONED_LIB"
            fi
        fi
        cd - > /dev/null
        ORT_LIB="$ORT_CAPI_DIR/libonnxruntime.so"
    else
        ORT_LIB=""
    fi
    
    echo -e "${GREEN}✓${NC} Python environment ready"
    echo "  Python: $(which python3)"
    echo "  Venv: $VENV_DIR"
    if [ -n "$ORT_LIB" ]; then
        echo "  ONNX Runtime: $ORT_LIB"
    fi
    echo ""
    
    # Build native extension if needed
    if [ "$BACKEND" = "torch-native" ]; then
        echo -e "${YELLOW}→${NC} Building native extension against venv..."
        
        # Force maturin to use our venv Python
        export PYO3_PYTHON="$VENV_DIR/bin/python3"
        
        cd daemon/native_ml
        
        # Clean previous builds
        rm -rf target/wheels/*.whl 2>/dev/null || true
        
        echo "  Building with Python: $PYO3_PYTHON"
        maturin build --release --interpreter "$PYO3_PYTHON"
        
        # Install into venv
        "$VENV_DIR/bin/pip" install target/wheels/*.whl --force-reinstall
        
        cd ../..
        echo -e "${GREEN}✓${NC} Native extension built and installed"
        echo ""
    fi
    
    deactivate
fi

# Build daemon
echo -e "${YELLOW}→${NC} Building daemon (this may take a few minutes)..."
echo "  Features: $CARGO_FEATURES"

# Set build env for native backend
if [ "$BACKEND" = "torch-native" ]; then
    export PYO3_PYTHON="$DATA_DIR/venv/bin/python3"
    export VIRTUAL_ENV="$DATA_DIR/venv"
    echo "  Building with Python: $PYO3_PYTHON"
fi

cargo build --release --features "$CARGO_FEATURES"
echo -e "${GREEN}✓${NC} Daemon built"
echo ""

# Install binary
echo -e "${YELLOW}→${NC} Installing binary..."
cp target/release/doormand "$INSTALL_DIR/bin/"
chmod +x "$INSTALL_DIR/bin/doormand"
echo -e "${GREEN}✓${NC} Binary installed: $INSTALL_DIR/bin/doormand"
echo ""

# Download models
echo -e "${YELLOW}→${NC} Downloading ML models..."
if [ ! -f "$DATA_DIR/models/blazeface.onnx" ]; then
    echo "  Downloading models to $DATA_DIR/models/"
    # TODO: Implement model download
    echo -e "${YELLOW}  Note: Model download not yet implemented${NC}"
    echo -e "${YELLOW}  Please manually copy models to $DATA_DIR/models/${NC}"
else
    echo -e "${GREEN}✓${NC} Models already present"
fi
echo ""

# Generate config
echo -e "${YELLOW}→${NC} Generating configuration..."
cat > "$CONFIG_DIR/doorman.toml" << EOF
# Doorman Configuration
# Generated by installer

[ml]
backend = "$BACKEND"
device = "$([ "$GPU_TYPE" != "none" ] && echo "cuda" || echo "cpu")"
models_dir = "$DATA_DIR/models"
confidence_threshold = 0.7
liveness_threshold = 0.5
similarity_threshold = 0.6

[camera]
backend = "$CAMERA"
device = "/dev/video0"
width = 1280
height = 720
fps = 30

[daemon]
socket_path = "$SOCKET_PATH"
data_dir = "$DATA_DIR"
user_mode = $([ "$SYSTEMD_TYPE" = "user" ] && echo "true" || echo "false")

[storage]
embeddings_path = "$DATA_DIR/embeddings"

[logging]
level = "info"
EOF

echo -e "${GREEN}✓${NC} Config created: $CONFIG_DIR/doorman.toml"
echo ""

# Create systemd service
echo -e "${YELLOW}→${NC} Creating systemd service..."

if [ "$SYSTEMD_TYPE" = "user" ]; then
    cat > "$SERVICE_DIR/doormand.service" << EOF
[Unit]
Description=Doorman Face Authentication Daemon
After=graphical-session.target

[Service]
Type=simple
ExecStart=$INSTALL_DIR/bin/doormand --user --config $CONFIG_DIR/doorman.toml
Environment="VIRTUAL_ENV=$DATA_DIR/venv"
Environment="PATH=$DATA_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
$([ "$GPU_TYPE" = "amd" ] && echo 'Environment="HSA_OVERRIDE_GFX_VERSION=11.0.1"')
$([ -n "$ORT_CAPI_DIR" ] && echo "Environment=\"LD_LIBRARY_PATH=$ORT_CAPI_DIR:\$LD_LIBRARY_PATH\"")
$([ -n "$ORT_LIB" ] && echo "Environment=\"ORT_DYLIB_PATH=$ORT_LIB\"")
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=default.target
EOF
else
    cat > "$SERVICE_DIR/doormand.service" << EOF
[Unit]
Description=Doorman Face Authentication Daemon
After=multi-user.target

[Service]
Type=simple
ExecStart=$INSTALL_DIR/bin/doormand --config $CONFIG_DIR/doorman.toml
Environment="VIRTUAL_ENV=$DATA_DIR/venv"
Environment="PATH=$DATA_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
$([ "$GPU_TYPE" = "amd" ] && echo 'Environment="HSA_OVERRIDE_GFX_VERSION=11.0.1"')
$([ -n "$ORT_CAPI_DIR" ] && echo "Environment=\"LD_LIBRARY_PATH=$ORT_CAPI_DIR:\$LD_LIBRARY_PATH\"")
$([ -n "$ORT_LIB" ] && echo "Environment=\"ORT_DYLIB_PATH=$ORT_LIB\"")
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF
fi

echo -e "${GREEN}✓${NC} Service created: $SERVICE_DIR/doormand.service"
echo ""

# Reload systemd
echo -e "${YELLOW}→${NC} Reloading systemd..."
if [ "$SYSTEMD_TYPE" = "user" ]; then
    systemctl --user daemon-reload
else
    systemctl daemon-reload
fi
echo -e "${GREEN}✓${NC} Systemd reloaded"
echo ""

# Installation complete
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║               Installation Complete! 🎉                        ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo ""
echo -e "  1. Copy ML models to: ${YELLOW}$DATA_DIR/models/${NC}"
echo -e "     Required files: blazeface.onnx, liveness.onnx, mobilefacenet.onnx"
echo ""
echo -e "  2. Start the daemon:"
if [ "$SYSTEMD_TYPE" = "user" ]; then
    echo -e "     ${GREEN}systemctl --user start doormand${NC}"
    echo -e "     ${GREEN}systemctl --user enable doormand${NC}  (start on login)"
else
    echo -e "     ${GREEN}sudo systemctl start doormand${NC}"
    echo -e "     ${GREEN}sudo systemctl enable doormand${NC}  (start on boot)"
fi
echo ""
echo -e "  3. Check status:"
if [ "$SYSTEMD_TYPE" = "user" ]; then
    echo -e "     ${GREEN}systemctl --user status doormand${NC}"
    echo -e "     ${GREEN}journalctl --user -u doormand -f${NC}  (logs)"
else
    echo -e "     ${GREEN}sudo systemctl status doormand${NC}"
    echo -e "     ${GREEN}sudo journalctl -u doormand -f${NC}  (logs)"
fi
echo ""
echo -e "  4. Use the CLI:"
echo -e "     ${GREEN}doorman preview${NC}         - Live camera preview"
echo -e "     ${GREEN}doorman add-face <name>${NC} - Register new user"
echo -e "     ${GREEN}doorman list${NC}            - List registered users"
echo ""
echo -e "${YELLOW}Documentation: https://github.com/yourusername/doorman${NC}"
echo ""
