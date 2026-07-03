#!/bin/sh
# Stop the alpha web dashboard (launchd-managed or manual).
cd "$(dirname "$0")"
PID_FILE=".web.pid"
JOB="com.gangwu.alpha.dashboard"

if launchctl list 2>/dev/null | grep -q "$JOB"; then
    launchctl bootout "gui/$(id -u)/$JOB"
    echo "unloaded launchd dashboard job (re-enable with scripts/install_launchd.sh or launchctl bootstrap)"
    exit 0
fi

if [ ! -f "$PID_FILE" ]; then
    echo "no pid file — not running?"
    exit 0
fi
PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "stopped web dashboard (pid $PID)"
else
    echo "process $PID not running"
fi
rm -f "$PID_FILE"
