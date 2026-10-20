#!/usr/bin/env bash
# Only this project's monitor; never signals processes or reserves GPU memory.
set -u
ROOT=${LMM_REBUILD_ROOT:-/cluster/home/gaomingju/workspace/lmm/lmm}
mkdir -p "$ROOT/logs"
exec >>"$ROOT/logs/resource_monitor.log" 2>&1
while true; do
  date -Is
  nvidia-smi -i 7 --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
  awk '/MemAvailable|Shmem:/ {print}' /proc/meminfo
  df -h "$ROOT" /dev/shm
  tail -3 "$ROOT/runtime/rebuild_20261020/events.log" 2>/dev/null
  sleep 300
done
