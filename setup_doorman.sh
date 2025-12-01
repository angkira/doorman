#!/bin/bash
# Complete doorman setup script
# Builds, installs, and configures doorman user service

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚪 Doorman Face Authentication Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Error: Must run from doorman project root"
    exit 1
fi

# 1. Build
echo "📦 Step 1/5: Building project..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cargo build --release
echo "✅ Build complete"
echo ""

# 2. Install user service
echo "🔧 Step 2/5: Installing user service..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./install_user_service.sh
echo ""

# 3. Install models
echo "🤖 Step 3/5: Installing ML models..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "/var/lib/doorman/models" ] && [ "$(ls -A /var/lib/doorman/models 2>/dev/null)" ]; then
    echo "✅ Models already installed"
else
    echo "Installing models..."
    doorman models install || {
        echo "⚠️  Model installation failed, but continuing..."
        echo "   You can install models later with: doorman models install"
    }
fi
echo ""

# 4. Run tests
echo "🧪 Step 4/5: Running tests..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./test_user_service.sh
echo ""

# 5. Next steps
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Step 5/5: Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Enroll your face:"
echo "   $ doorman enroll $USER"
echo ""
echo "2. Test the preview:"
echo "   $ doorman preview"
echo ""
echo "3. Configure PAM (for login/sudo/lock):"
echo "   See README.md for PAM configuration"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📚 Documentation:"
echo "   - User service architecture: USER_SERVICE.md"
echo "   - Full documentation: README.md"
echo ""
echo "🔍 Useful Commands:"
echo "   - View logs: journalctl --user -u doormand-user -f"
echo "   - Check status: doorman status"
echo "   - List users: doorman users"
echo "   - Restart daemon: systemctl --user restart doormand-user"
echo ""
echo "🎉 Your face authentication system is ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
