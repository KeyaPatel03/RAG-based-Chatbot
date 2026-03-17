#!/bin/bash
set -e

# Detect Configs
PROJECT_DIR=$(pwd)
PYTHON_EXEC=$(which python3)
USER_SYSTEMD_DIR="$HOME/.config/systemd/user"

echo "=== Track2College Auto-Scheduler Setup ==="
echo "Project Dir: $PROJECT_DIR"
echo "Python Exec: $PYTHON_EXEC"
echo "Systemd Dir: $USER_SYSTEMD_DIR"
echo "=========================================="

# Create systemd dir if needed
mkdir -p "$USER_SYSTEMD_DIR"

# process template
sed -e "s|{{PROJECT_DIR}}|$PROJECT_DIR|g" \
    -e "s|{{PYTHON_EXEC}}|$PYTHON_EXEC|g" \
    deployment/track2college.service.template > "$USER_SYSTEMD_DIR/track2college-pipeline.service"

# Copy timer
cp deployment/track2college.timer "$USER_SYSTEMD_DIR/track2college-pipeline.timer"

# Reload and Enable
systemctl --user daemon-reload
systemctl --user enable --now track2college-pipeline.timer

# Enable Linger (allows running without active session)
if loginctl enable-linger "$USER" 2>/dev/null; then
    echo "✅ Persistence enabled (Linger=yes)"
else
    echo "⚠️  Could not enable persistence (loginctl failed). You may need to stay logged in."
fi

echo ""
echo "✅ Setup Complete!"
echo "   - Service created: $USER_SYSTEMD_DIR/track2college-pipeline.service"
echo "   - Timer enabled:   track2college-pipeline.timer"
echo ""
echo "Next run scheduled for:"
systemctl --user list-timers --all | grep track2college
