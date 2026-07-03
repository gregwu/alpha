#!/bin/sh
# Stop the alpha web dashboard started by start.sh.
cd "$(dirname "$0")"
PID_FILE=".web.pid"

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
