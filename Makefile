.PHONY: build install test clean uninstall dev

# Build all components
build:
	@echo "🔨 Building Rust components..."
	cargo build --release
	@echo "✓ Build complete"

# Install Python CLI (development mode)
install-cli:
	@echo "📦 Installing Python CLI..."
	uv pip install -e .
	@echo "✓ CLI installed"

# Full installation (requires root)
install: build
	@echo "🚀 Installing doorman..."
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "❌ Error: Installation requires root privileges"; \
		echo "Run: sudo make install"; \
		exit 1; \
	fi
	@echo "Installing binaries..."
	cp target/release/libpam_doorman.so /usr/lib/x86_64-linux-gnu/security/pam_doorman.so
	cp target/release/doormand /usr/local/bin/doormand
	chmod 644 /usr/lib/x86_64-linux-gnu/security/pam_doorman.so
	chmod 755 /usr/local/bin/doormand
	@echo "Creating directories..."
	mkdir -p /var/lib/doorman/models
	chmod 700 /var/lib/doorman
	chmod 755 /var/lib/doorman/models
	@echo "Installing systemd service..."
	cp doormand.service /etc/systemd/system/
	systemctl daemon-reload
	systemctl enable doormand
	systemctl start doormand
	@echo "✓ Installation complete"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Download models to /var/lib/doorman/models/ (see MODELS.md)"
	@echo "  2. Enroll your face: sudo doorman enroll"
	@echo "  3. Configure PAM: sudo doorman setup"

# Development: build and run daemon locally
dev:
	@echo "🔧 Starting daemon in development mode..."
	RUST_LOG=debug cargo run --release --bin doormand

# Run tests
test:
	@echo "🧪 Running tests..."
	cargo test --all
	@echo "Testing IPC connection..."
	@if systemctl is-active --quiet doormand; then \
		echo '{"type":"status"}' | nc -U /run/doorman.sock || echo "Daemon not responding"; \
	else \
		echo "Daemon not running (start with: sudo systemctl start doormand)"; \
	fi

# Run tests with video support
test-video:
	@echo "🎥 Running tests with video support..."
	cargo test --features video --all

# Run integration tests
test-e2e:
	@echo "🔗 Running E2E tests..."
	cargo test --test e2e_test

# Run Python tests
test-python:
	@echo "🐍 Running Python tests..."
	pytest src/doorman/test_cli.py -v

# Run all tests (unit + integration + Python)
test-all: test test-e2e test-python
	@echo "✅ All tests complete"

# Build with GPU support
build-gpu:
	@echo "🎮 Building with GPU support..."
	cargo build --release --features gpu

# Test with GPU (ROCm for AMD)
test-gpu-rocm:
	@echo "Testing ROCm GPU configuration..."
	cargo test --test config_tests::test_config_gpu_rocm

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	cargo clean
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "✓ Clean complete"

# Uninstall (requires root)
uninstall:
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "❌ Error: Uninstallation requires root privileges"; \
		echo "Run: sudo make uninstall"; \
		exit 1; \
	fi
	@echo "🗑️  Uninstalling doorman..."
	systemctl stop doormand || true
	systemctl disable doormand || true
	rm -f /usr/lib/x86_64-linux-gnu/security/pam_doorman.so
	rm -f /usr/local/bin/doormand
	rm -f /etc/systemd/system/doormand.service
	systemctl daemon-reload
	@echo "✓ Uninstalled (data preserved at /var/lib/doorman)"
	@echo "To remove data: sudo rm -rf /var/lib/doorman"

# Check system dependencies
check-deps:
	@echo "🔍 Checking dependencies..."
	@command -v rustc >/dev/null 2>&1 || echo "❌ rustc not found"
	@command -v cargo >/dev/null 2>&1 || echo "❌ cargo not found"
	@command -v gcc >/dev/null 2>&1 || echo "❌ gcc not found"
	@command -v uv >/dev/null 2>&1 || echo "❌ uv not found"
	@test -f /usr/include/security/pam_appl.h || echo "❌ libpam0g-dev not found"
	@echo "✓ Dependency check complete"

# Quick status check
status:
	@echo "📊 doorman status:"
	@systemctl is-active doormand && echo "Daemon: ✓ running" || echo "Daemon: ✗ stopped"
	@test -f /usr/local/bin/doormand && echo "Binary: ✓ installed" || echo "Binary: ✗ not installed"
	@test -f /usr/lib/x86_64-linux-gnu/security/pam_doorman.so && echo "PAM module: ✓ installed" || echo "PAM module: ✗ not installed"
	@test -d /var/lib/doorman/models && echo "Models dir: ✓ exists" || echo "Models dir: ✗ missing"

# Show logs
logs:
	@journalctl -u doormand -f

# Help
help:
	@echo "doorman Makefile commands:"
	@echo ""
	@echo "Build:"
	@echo "  make build        - Build Rust components"
	@echo "  make build-gpu    - Build with GPU support"
	@echo "  make install      - Full installation (requires sudo)"
	@echo "  make install-cli  - Install Python CLI only"
	@echo "  make dev          - Run daemon in development mode"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run unit tests"
	@echo "  make test-video   - Run tests with video support"
	@echo "  make test-e2e     - Run integration tests"
	@echo "  make test-python  - Run Python tests"
	@echo "  make test-all     - Run all tests"
	@echo "  make test-gpu-rocm- Test ROCm GPU config"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make uninstall    - Uninstall doorman (requires sudo)"
	@echo "  make check-deps   - Check system dependencies"
	@echo "  make status       - Show installation status"
	@echo "  make logs         - Show daemon logs (live)"
	@echo "  make help         - Show this help"

