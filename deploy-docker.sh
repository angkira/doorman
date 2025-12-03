#!/bin/bash
# Deploy Doorman with Docker-based ML inference
# Complete setup: container + daemon + systemd service

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/doorman}"
DATA_DIR="${DATA_DIR:-$HOME/.local/share/doorman}"
MODELS_DIR="$DATA_DIR/models"

echo "=============================================="
echo "   Doorman Docker Deployment"
echo "=============================================="
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Install dir: $INSTALL_DIR"
echo "Data dir: $DATA_DIR"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first:"
    echo "   https://docs.docker.com/engine/install/"
    exit 1
fi

echo "✓ Docker found: $(docker --version)"

# Create directories
mkdir -p "$INSTALL_DIR"/{bin,config}
mkdir -p "$DATA_DIR"/{models,embeddings}
mkdir -p "$HOME/.config/systemd/user"

# Step 1: Build Docker image
echo ""
echo "[1/6] Building Docker inference container..."
echo "----------------------------------------------"
cd "$PROJECT_ROOT/docker"
docker compose build
echo "✓ Container built"

# Step 2: Download models if needed
echo ""
echo "[2/6] Checking ONNX models..."
echo "----------------------------------------------"
if [ ! -f "$MODELS_DIR/blazeface.onnx" ]; then
    echo "Models not found. Downloading..."
    python3 "$PROJECT_ROOT/tools/download_models.py" --output "$MODELS_DIR" || {
        echo "⚠️  Model download failed. Please download manually:"
        echo "    BlazeFace: https://github.com/onnx/models/tree/main/validated/vision/body_analysis/ultraface"
        echo "    MobileFaceNet: https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface"
        echo "    Place in: $MODELS_DIR"
    }
else
    echo "✓ Models found"
fi

# Step 3: Build Rust daemon
echo ""
echo "[3/6] Building Rust daemon with Socket backend..."
echo "----------------------------------------------"
cd "$PROJECT_ROOT"
cargo build --release --features backend-socket,camera-gstreamer
echo "✓ Daemon built"

# Step 4: Install binaries
echo ""
echo "[4/6] Installing binaries..."
echo "----------------------------------------------"
cp target/release/doormand "$INSTALL_DIR/bin/"
cp doorman-socket.toml "$INSTALL_DIR/config/doorman.toml"

# Update paths in config
sed -i "s|~/.local/share/doorman|$DATA_DIR|g" "$INSTALL_DIR/config/doorman.toml"
sed -i "s|/run/user/1000/doorman.sock|/run/user/$UID/doorman.sock|g" "$INSTALL_DIR/config/doorman.toml"

echo "✓ Installed to $INSTALL_DIR"

# Step 5: Create systemd services
echo ""
echo "[5/6] Creating systemd services..."
echo "----------------------------------------------"

# Docker container service
cat > "$HOME/.config/systemd/user/doorman-inference.service" << EOF
[Unit]
Description=Doorman ML Inference Container (ONNX+ROCm)
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$PROJECT_ROOT/docker
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
Restart=on-failure

[Install]
WantedBy=default.target
EOF

# Daemon service (depends on container)
cat > "$HOME/.config/systemd/user/doormand.service" << EOF
[Unit]
Description=Doorman Face Recognition Daemon
Requires=doorman-inference.service
After=doorman-inference.service
PartOf=doorman-inference.service

[Service]
Type=simple
ExecStart=$INSTALL_DIR/bin/doormand --user --config $INSTALL_DIR/config/doorman.toml
Restart=always
RestartSec=5

# Wait for container socket to be ready
ExecStartPre=/bin/bash -c 'for i in {1..30}; do [ -S /tmp/doorman-ml.sock ] && break || sleep 2; done'

Environment="PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
Environment="XDG_RUNTIME_DIR=/run/user/%U"

[Install]
WantedBy=default.target
EOF

echo "✓ Systemd services created"

# Step 6: Enable and start services
echo ""
echo "[6/6] Starting services..."
echo "----------------------------------------------"

systemctl --user daemon-reload
systemctl --user enable doorman-inference.service doormand.service
systemctl --user start doorman-inference.service

echo ""
echo "Waiting for inference container socket..."
for i in {1..30}; do
    if [ -S /tmp/doorman-ml.sock ]; then
        echo "✓ Inference container ready (socket created)!"
        break
    fi
    sleep 2
    if [ $i -eq 30 ]; then
        echo "❌ Timeout waiting for socket. Check: docker compose logs"
        exit 1
    fi
done

systemctl --user start doormand.service

echo ""
echo "=============================================="
echo "✅ Deployment Complete!"
echo "=============================================="
echo ""
echo "Services:"
echo "  • Inference: systemctl --user status doorman-inference"
echo "  • Daemon:    systemctl --user status doormand"
echo ""
echo "Logs:"
echo "  • Container: docker compose -f $PROJECT_ROOT/docker/docker-compose.yml logs -f"
echo "  • Daemon:    journalctl --user -u doormand -f"
echo ""
echo "Commands:"
echo "  • doorman enroll <username>  - Enroll new user"
echo "  • doorman verify <username>  - Test recognition"
echo "  • doorman preview            - Live camera preview"
echo ""
echo "Configuration: $INSTALL_DIR/config/doorman.toml"
echo ""
echo "Expected performance: 50-60 FPS on AMD Radeon 780M iGPU! 🚀"
echo ""
