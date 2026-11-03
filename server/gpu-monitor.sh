#!/usr/bin/env bash
# 1 Hz GPU sampler with an epoch timestamp, so analyze.py can cut out each run's window.
# Usage: server/gpu-monitor.sh start <label> | stop
# Output: results/<label>/gpu.csv  columns: epoch,mem_mib,util_pct,temp_c,sm_clock_mhz,power_w
set -uo pipefail
cd "$(dirname "$0")/.."
source config.env
case "${1:-}" in
  start)
    label="${2:?label}"
    mkdir -p "$RESULTS_DIR/$label" logs
    out="$RESULTS_DIR/$label/gpu.csv"
    (
      while true; do
        s=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw \
              --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        echo "$(date +%s.%N | cut -c1-14),$s"
        sleep 1
      done
    ) >> "$out" 2>/dev/null &
    echo $! > logs/gpu-monitor.pid
    echo "monitor -> $out (pid $(cat logs/gpu-monitor.pid))"
    ;;
  stop)
    [ -f logs/gpu-monitor.pid ] && { kill "$(cat logs/gpu-monitor.pid)" 2>/dev/null; rm -f logs/gpu-monitor.pid; echo "monitor stopped"; }
    ;;
  *) echo "usage: $0 start <label> | stop"; exit 1 ;;
esac
