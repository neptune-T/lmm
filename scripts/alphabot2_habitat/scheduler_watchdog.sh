#!/bin/bash
# Keeps exactly one gpu_scheduler.py alive (tmux window "scheduler"). The scheduler itself holds a flock,
# so a duplicate start exits at once; a crashed scheduler is restarted within a minute and adopts the
# running collection / conversion / SFT / eval processes.
#   scheduler_watchdog.sh [SCHEDULE_JSON]
L=${LMM_REPO:-/workspace/lmm}
CFG=${1:-$L/configs/alphabot2_pi05/schedule.json}
LOGS=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['logs'])" $CFG)
exec 9> /dev/shm/lmm_scheduler_watchdog.lock
flock -n 9 || { echo "another watchdog is running"; exit 0; }
while true; do
  if ! pgrep -f "$L/scripts/alphabot2_habitat/gpu_scheduler.py" > /dev/null; then
    echo "$(date '+%F %T') starting gpu_scheduler.py" | tee -a $LOGS/watchdog.log
    python3 $L/scripts/alphabot2_habitat/gpu_scheduler.py $CFG >> $LOGS/scheduler.log 2>&1
    echo "$(date '+%F %T') gpu_scheduler.py exited rc=$?" | tee -a $LOGS/watchdog.log
  fi
  sleep 60
done
