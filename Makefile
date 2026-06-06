.PHONY: build build-daemon build-pam build-rocm build-cuda models test-docker \
        install install-bins install-pam install-systemd install-config \
        uninstall pam-instructions check-deps status logs test clean help

# ---------------------------------------------------------------------------
# Lean CPU build (default feature set = backend-ort + camera-mock +
# camera-ffmpeg + preview). For a REAL Linux login/lock-screen unlock we add the
# native camera backends. Override CAMERA_FEATURES to taste.
# ---------------------------------------------------------------------------
PREFIX            ?= /usr
BINDIR            ?= $(PREFIX)/bin
DATADIR           ?= /var/lib/doorman
MODELSDIR         ?= $(DATADIR)/models
CONFDIR           ?= /etc/doorman
SYSTEMD_DIR       ?= /etc/systemd/system
SERVICE_USER      ?= doorman

# Camera backends compiled into the daemon for real Linux use.
# camera-v4l2 (nokhwa/V4L2) + camera-gstreamer (PipeWire) + camera-ffmpeg + mock.
CAMERA_FEATURES   ?= camera-v4l2,camera-gstreamer
# Daemon features: CPU ONNX Runtime + cameras (+ mock fallback). No preview GUI
# in the service binary build to keep the dependency tree lean.
DAEMON_FEATURES   ?= backend-ort,camera-mock,camera-ffmpeg,$(CAMERA_FEATURES)

# Detect the system PAM security directory (Debian/Ubuntu multiarch vs others).
PAM_DIR := $(shell \
	for d in /usr/lib/$(shell uname -m)-linux-gnu/security \
	         /usr/lib/x86_64-linux-gnu/security \
	         /usr/lib/aarch64-linux-gnu/security \
	         /lib/security /usr/lib/security /usr/lib64/security; do \
		[ -d "$$d" ] && echo "$$d" && break; \
	done)
PAM_DIR := $(or $(PAM_DIR),/lib/security)

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
build: build-daemon build-pam
	@echo "✓ Build complete (daemon features: $(DAEMON_FEATURES))"

# Daemon + doorman CLI (release, lean CPU + native cameras).
build-daemon:
	@echo "🔨 Building doormand + doorman CLI ($(DAEMON_FEATURES))..."
	cargo build --release -p doormand --no-default-features --features "$(DAEMON_FEATURES)"

# PAM module is Linux-only; on macOS it compiles to an empty cdylib.
build-pam:
	@echo "🔨 Building libpam_doorman.so..."
	cargo build --release -p pam_doorman

# Optional GPU builds (Linux execution providers).
build-rocm:
	cargo build --release -p doormand --no-default-features --features "backend-ort-rocm,camera-mock,camera-ffmpeg,$(CAMERA_FEATURES)"

build-cuda:
	cargo build --release -p doormand --no-default-features --features "backend-ort-cuda,camera-mock,camera-ffmpeg,$(CAMERA_FEATURES)"

# Fetch / verify ONNX models into data/models and the runtime dir.
models:
	@echo "📥 Fetching models..."
	./scripts/fetch_models.sh

# ---------------------------------------------------------------------------
# Install (system service). Requires root.
# ---------------------------------------------------------------------------
install: build install-bins install-pam install-config install-systemd
	@echo ""
	@echo "✓ Installation complete."
	@echo ""
	@echo "Next steps:"
	@echo "  1. Fetch models (as the service user):"
	@echo "       sudo -u $(SERVICE_USER) ./scripts/fetch_models.sh   # or: make models"
	@echo "       (models must end up in $(MODELSDIR))"
	@echo "  2. Enroll your face:   doorman enroll \$$USER"
	@echo "  3. Test it works:      doorman test \$$USER   # MUST pass before touching PAM"
	@echo "  4. Configure PAM:      make pam-instructions"
	@echo ""
	@$(MAKE) --no-print-directory pam-instructions

install-bins:
	@echo "📦 Installing binaries to $(BINDIR)..."
	install -d $(BINDIR)
	install -m 0755 target/release/doormand $(BINDIR)/doormand
	@if [ -f target/release/doorman ]; then \
		install -m 0755 target/release/doorman $(BINDIR)/doorman; \
		echo "  ✓ doorman CLI installed"; \
	else \
		echo "  ⚠ doorman CLI not built (target/release/doorman missing) — skipping"; \
	fi
	@echo "👤 Ensuring service user '$(SERVICE_USER)' exists..."
	@id -u $(SERVICE_USER) >/dev/null 2>&1 || \
		useradd --system --no-create-home --shell /usr/sbin/nologin --groups video $(SERVICE_USER)
	@id -nG $(SERVICE_USER) | grep -qw video || usermod -aG video $(SERVICE_USER) || true
	@echo "📁 Creating $(DATADIR) and $(MODELSDIR)..."
	install -d -o $(SERVICE_USER) -g $(SERVICE_USER) -m 0750 $(DATADIR)
	install -d -o $(SERVICE_USER) -g $(SERVICE_USER) -m 0750 $(MODELSDIR)

install-pam:
	@echo "🔐 Installing PAM module to $(PAM_DIR)..."
	install -d $(PAM_DIR)
	install -m 0644 target/release/libpam_doorman.so $(PAM_DIR)/pam_doorman.so
	@echo "  ✓ $(PAM_DIR)/pam_doorman.so"

install-config:
	@echo "⚙️  Installing config to $(CONFDIR)..."
	install -d $(CONFDIR)
	@if [ -f $(CONFDIR)/doorman.toml ]; then \
		echo "  • $(CONFDIR)/doorman.toml exists — leaving it untouched"; \
	else \
		install -m 0644 packaging/doorman.toml.example $(CONFDIR)/doorman.toml; \
		echo "  ✓ wrote default $(CONFDIR)/doorman.toml"; \
	fi

install-systemd:
	@echo "🧩 Installing systemd units..."
	install -m 0644 packaging/systemd/doormand.service $(SYSTEMD_DIR)/doormand.service
	@# Ship the user unit for desktop/dev use too (not enabled automatically).
	install -d /usr/lib/systemd/user
	install -m 0644 packaging/systemd/doormand-user.service /usr/lib/systemd/user/doormand.service
	systemctl daemon-reload
	@echo "  ✓ Enable with: sudo systemctl enable --now doormand.service"

# ---------------------------------------------------------------------------
# /etc/pam.d guidance — we DO NOT edit PAM configs automatically (lockout risk).
# ---------------------------------------------------------------------------
pam-instructions:
	@echo "──────────────────────────────────────────────────────────────────────"
	@echo " PAM configuration (MANUAL — do NOT skip the safety step)"
	@echo "──────────────────────────────────────────────────────────────────────"
	@echo ""
	@echo " ⚠  KEEP A ROOT SHELL OPEN in a separate terminal while editing PAM."
	@echo "    A mistake in these files can lock you out of sudo AND the GUI."
	@echo "    Test with that root shell still open before logging out."
	@echo ""
	@echo " Add this line as the FIRST auth line in the relevant file(s):"
	@echo ""
	@echo "        auth    sufficient    pam_doorman.so"
	@echo ""
	@echo " 'sufficient' = a face match logs you in; anything else (no match,"
	@echo " daemon down, timeout) falls through to your normal password prompt."
	@echo ""
	@echo " Where to add it (pick what you use):"
	@echo "   • sudo:                 /etc/pam.d/sudo        (above @include common-auth)"
	@echo "   • GNOME login/unlock:   /etc/pam.d/gdm-password (above @include common-auth)"
	@echo "   • screen lock (GNOME):  /etc/pam.d/gdm-password covers it"
	@echo "   • all PAM auth at once: /etc/pam.d/common-auth  (top of file) — broad, riskier"
	@echo ""
	@echo " Example /etc/pam.d/sudo:"
	@echo "        #%PAM-1.0"
	@echo "        auth       sufficient   pam_doorman.so"
	@echo "        @include common-auth"
	@echo "        ..."
	@echo ""
	@echo " To undo: remove the 'auth sufficient pam_doorman.so' line you added."
	@echo "──────────────────────────────────────────────────────────────────────"

# ---------------------------------------------------------------------------
# Uninstall. Requires root. Preserves enrolled data unless you remove $(DATADIR).
# ---------------------------------------------------------------------------
uninstall:
	@echo "🗑️  Uninstalling doorman..."
	-systemctl disable --now doormand.service 2>/dev/null || true
	rm -f $(SYSTEMD_DIR)/doormand.service
	rm -f /usr/lib/systemd/user/doormand.service
	-systemctl daemon-reload 2>/dev/null || true
	rm -f $(PAM_DIR)/pam_doorman.so
	rm -f $(BINDIR)/doormand $(BINDIR)/doorman
	@echo "✓ Uninstalled."
	@echo "  • REMOVE the 'auth sufficient pam_doorman.so' lines you added to /etc/pam.d/*"
	@echo "  • Data preserved at $(DATADIR) (remove with: sudo rm -rf $(DATADIR))"
	@echo "  • Service user '$(SERVICE_USER)' left in place (remove with: sudo userdel $(SERVICE_USER))"

# ---------------------------------------------------------------------------
# Dev / diagnostics
# ---------------------------------------------------------------------------
test:
	@echo "🧪 Running tests..."
	cargo test --workspace

# Build the Ubuntu 24.04 test image and run the Linux validation harness
# (enroll -> genuine match -> impostor reject) end-to-end. Set PLATFORM=linux/amd64
# for the true x86_64 path (emulated on ARM hosts). Requires Docker.
test-docker:
	@echo "🐳 Building + running Ubuntu container test..."
	scripts/test_ubuntu.sh

clean:
	cargo clean

check-deps:
	@echo "🔍 Checking build dependencies..."
	@command -v cargo >/dev/null 2>&1 && echo "  ✓ cargo" || echo "  ❌ cargo (install rustup)"
	@command -v clang >/dev/null 2>&1 && echo "  ✓ clang"  || echo "  ❌ clang (apt install clang)"
	@test -f /usr/include/security/pam_appl.h && echo "  ✓ libpam0g-dev" || echo "  ❌ libpam0g-dev"
	@{ pkg-config --exists openssl 2>/dev/null || test -f /usr/include/openssl/ssl.h; } && echo "  ✓ libssl-dev" || echo "  ❌ libssl-dev (apt install libssl-dev)"
	@command -v pkg-config >/dev/null 2>&1 && echo "  ✓ pkg-config" || echo "  ❌ pkg-config"
	@echo "  PAM install dir detected: $(PAM_DIR)"

status:
	@systemctl is-active doormand.service >/dev/null 2>&1 && echo "Daemon: ✓ running" || echo "Daemon: ✗ stopped"
	@test -f $(BINDIR)/doormand && echo "Binary: ✓ $(BINDIR)/doormand" || echo "Binary: ✗ not installed"
	@test -f $(PAM_DIR)/pam_doorman.so && echo "PAM:    ✓ $(PAM_DIR)/pam_doorman.so" || echo "PAM:    ✗ not installed"
	@test -d $(MODELSDIR) && echo "Models: ✓ $(MODELSDIR)" || echo "Models: ✗ missing"

logs:
	@journalctl -u doormand.service -f

help:
	@echo "doorman Makefile — lean CPU build"
	@echo ""
	@echo "Build:"
	@echo "  make build            Build daemon + CLI + PAM (CPU, native cameras)"
	@echo "  make build-rocm       Build daemon with AMD ROCm execution provider"
	@echo "  make build-cuda       Build daemon with NVIDIA CUDA execution provider"
	@echo "  make models           Fetch/verify ONNX models"
	@echo ""
	@echo "Install (sudo):"
	@echo "  make install          Build + install binaries, PAM module, config, systemd"
	@echo "  make pam-instructions Print the exact /etc/pam.d edit (manual, on purpose)"
	@echo "  make uninstall        Remove installed files (keeps enrolled data)"
	@echo ""
	@echo "Diagnostics:"
	@echo "  make check-deps | status | logs | test | clean"
	@echo ""
	@echo "Overridable vars: PREFIX BINDIR DATADIR MODELSDIR CONFDIR CAMERA_FEATURES"
