#!/bin/sh
# Start (or restart) the alpha web dashboard.
# If the launchd keepalive job is installed (scripts/install_launchd.sh),
# it owns the process; otherwise fall back to a manual nohup instance.
set -e
cd "$(dirname "$0")"

PORT="${1:-8100}"
PID_FILE=".web.pid"
JOB="com.gangwu.alpha.dashboard"

if launchctl list 2>/dev/null | grep -q "$JOB"; then
    launchctl kickstart -k "gui/$(id -u)/$JOB"
    echo "restarted launchd-managed dashboard at http://localhost:8100"
    exit 0
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "already running (pid $(cat "$PID_FILE")) — ./stop.sh first"
    exit 1
fi

# Build the frontend if the source is newer than the last build
if [ ! -d web/frontend/dist ] || \
   [ -n "$(find web/frontend/src web/frontend/index.html -newer web/frontend/dist -print -quit 2>/dev/null)" ]; then
    echo "building frontend..."
    (cd web/frontend && npm run build)
fi

mkdir -p logs
nohup python3 -m uvicorn web.backend.main:app --host 127.0.0.1 --port "$PORT" \
    > logs/web.log 2>&1 &
echo $! > "$PID_FILE"
sleep 1
if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "alpha dashboard running at http://localhost:$PORT (pid $(cat "$PID_FILE"), log: logs/web.log)"
else
    echo "failed to start — see logs/web.log"
    exit 1
fi
