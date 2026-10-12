#!/bin/bash
# Embodiment screen of the OVMM eval split, sharded by scene, detached (started by gpu_scheduler.py).
# Writes OUT/VAL_SCREEN_DONE when every shard has exited; the scheduler then runs the embodiment gate.
#   val_screen.sh RUNS GPU NWORKERS REPLICAS SPLIT LOGS [OUT]
R=$1 G=$2 NW=$3 REP=$4 SP=$5 LOGS=$6
S=$(cd "$(dirname "$0")" && pwd)
OUT=${7:-$R/emb_$SP}
mkdir -p $OUT
for w in $(seq 0 $((NW - 1))); do
  SCREEN=$R/feasibility/screen_$SP.json SPLIT=$SP REPLICAS=$REP GPU=$G WORKER=$w NWORKERS=$NW OUT=$OUT \
    bash $S/run_habitat.sh $S/emb_screen.py >> $LOGS/$(basename $OUT)_w$w.log 2>&1 &
done
wait
date '+%F %T' > $OUT/VAL_SCREEN_DONE
