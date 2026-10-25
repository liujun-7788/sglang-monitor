#!/bin/bash
# SGLang monitor dashboard start/stop helper.
# Usage: monitor_ctl.sh {start|stop|status} [dashboard-port]
# Optional env: SGLANG_MONITOR_PYTHON (default python3)
PY=${SGLANG_MONITOR_PYTHON:-python3}
D="$(cd "$(dirname "$0")" && pwd)"
case "$1" in
  start)
    if pgrep -f 'sglang_monitor.py' > /dev/null; then echo "already running"; exit 0; fi
    cd "$D" && setsid nohup "$PY" sglang_monitor.py ${2:+--port $2} > "$D/monitor.log" 2>&1 < /dev/null &
    sleep 3; curl -s -m 5 "http://127.0.0.1:${2:-7100}/health" && echo " started"
    ;;
  stop)
    pkill -f 'sglang_monitor.py' && echo stopped
    ;;
  status)
    pgrep -af 'sglang_monitor.py' | head -1 || echo "not running"
    curl -s -m 5 "http://127.0.0.1:${2:-7100}/health" && echo
    ;;
  *) echo "usage: $0 {start|stop|status} [port]";;
esac
