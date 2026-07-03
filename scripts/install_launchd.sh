#!/bin/sh
# Install the launchd jobs: daily pipeline (weekdays 18:30) and the
# dashboard keepalive (starts at login, auto-restarts on crash).
set -e
cd "$(dirname "$0")/.."
AGENTS="$HOME/Library/LaunchAgents"
UID_N=$(id -u)

# The keepalive job owns the dashboard from now on — stop any manual one
./stop.sh 2>/dev/null || true

for name in com.gangwu.alpha.pipeline com.gangwu.alpha.dashboard com.gangwu.alpha.trading; do
    cp "launchd/$name.plist" "$AGENTS/$name.plist"
    launchctl bootout "gui/$UID_N/$name" 2>/dev/null || true
    launchctl bootstrap "gui/$UID_N" "$AGENTS/$name.plist"
    echo "installed + loaded $name"
done

sleep 2
launchctl list | grep com.gangwu.alpha || true
echo "dashboard: http://localhost:8100  (managed by launchd)"
echo "pipeline : weekdays 18:30, log: logs/launchd_pipeline.log"
