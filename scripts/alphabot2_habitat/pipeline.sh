#!/bin/bash
# AlphaBot 2 OVMM restore -> qualification -> pilot -> continuous collection, with automatic gates.
# Each stage is skipped when its gate already passed (rerun after a fix); a failed gate stops here
# with BLOCKED in $RUNS/pipeline_status.jsonl. Gates: scripts/alphabot2_habitat/gates.py.
#     GPU=3 bash pipeline.sh            (detach with setsid nohup)
set -uo pipefail
L=${LMM_REPO:-/workspace/lmm}; R=${RUNTIME:-/workspace/runtime}; RUNS=$R/runs; S=$L/scripts/alphabot2_habitat
RUN="bash $S/run_habitat.sh"; GPU=${GPU:-3}; ST=$RUNS/pipeline_status.jsonl
mkdir -p $RUNS/logs $RUNS/lists
note() { echo "{\"time\": \"$(date '+%F %T')\", \"stage\": \"$1\", \"event\": \"$2\"}" >> $ST; echo "$(date '+%F %T') $1 $2"; }
passed() { grep -q "\"gate\": \"$1\", \"pass\": true" $ST 2>/dev/null; }
gate() {   # gate NAME ARGS...
  local name=$1; shift
  local out; out=$($RUN $S/gates.py $name "$@" 2>>$RUNS/logs/gates.err | grep '###GATE###' | sed 's/^###GATE###//')
  [ -n "$out" ] || out="{\"gate\": \"$name\", \"pass\": false, \"error\": \"gate produced no output\"}"
  echo "$out" >> $ST
  if echo "$out" | grep -q '"pass": true'; then note $name PASS; else note $name BLOCKED; exit 1; fi
}
par() {    # par N LOGNAME ENV... SCRIPT : N parallel shards (WORKER/NWORKERS), wait for all
  local n=$1 log=$2; shift 2
  for w in $(seq 0 $((n - 1))); do env WORKER=$w NWORKERS=$n "$@" > $RUNS/logs/${log}_w$w.log 2>&1 & done; wait
}

# 1. full geometric feasibility re-screen: train 6331 / val 133 / 38 of 38 train scenes
passed screen || { note screen start
  CUDA_VISIBLE_DEVICES=$GPU SPLITS=train,val OUT=$RUNS/feasibility $RUN $S/feasibility_screen.py > $RUNS/logs/screen.log 2>&1
  gate screen $RUNS/feasibility/summary.json $L/artifacts/alphabot2_habitat/feasibility_summary.json; }

# 2. navmesh regression: the documented radius-0.38 val audit (nav_qual defaults: 2 episodes x first 15 scenes)
passed navmesh || { note navmesh start; rm -rf $RUNS/nav_audit
  par 4 nav_audit GPU=$GPU SPLIT=val OUT=$RUNS/nav_audit $RUN $S/nav_qual.py
  gate navmesh $RUNS/nav_audit; }

# 3. runtime smoke on one episode the audit found reachable (both legs arrived)
passed smoke || { note smoke start
  SE=$(python3 -c "
import glob,json
for f in sorted(glob.glob('$RUNS/nav_audit/results_w*.jsonl')):
    for l in open(f):
        r=json.loads(l)
        if not r.get('engineering_error') and r.get('to_object',{}).get('arrived') and r.get('to_receptacle',{}).get('arrived'):
            print(r['episode_id']); raise SystemExit")
  note smoke "episode_$SE"
  GPU=$GPU SPLIT=val SMOKE_EPISODE=$SE SMOKE_OUT=$RUNS/smoke $RUN $S/smoke.py > $RUNS/logs/smoke.log 2>&1
  gate smoke $RUNS/smoke/smoke.json; }

# 4. episode lists from the re-screen (scene round-robin): qualification 5, pilot candidates, all feasible TRAIN
P=$S/pick_episodes.py
[ -s $RUNS/lists/collect_train.txt ] || python3 - <<PY
import subprocess
pick = lambda n, seed: subprocess.check_output(["python3", "$P", "$RUNS/feasibility/screen_train.json", str(n), str(seed)], text=True).strip().split(",")
q = pick(5, 1)
pilot = [e for e in pick(80, 2) if e not in q][:40]
rest = [e for e in pick(1000000, 0) if e not in q and e not in pilot]
open("$RUNS/lists/qual_train.txt", "w").write(",".join(q))
open("$RUNS/lists/pilot_candidates.txt", "w").write(",".join(pilot))
open("$RUNS/lists/collect_train.txt", "w").write("\n".join(pilot + rest) + "\n")
print("LISTS", len(q), len(pilot), len(pilot + rest))
PY

# 5. manipulation qualification on feasible TRAIN episodes: forced-left, forced-right, auto
passed manip || { note manip start; rm -rf $RUNS/manip_qual
  par 3 manip GPU=$GPU SPLIT=train MODES=left,right,auto EPISODES=$(cat $RUNS/lists/qual_train.txt) OUT=$RUNS/manip_qual $RUN $S/manip_qual.py
  gate manip $RUNS/manip_qual; }

# 6. snapshot/restore (tolerance gate) on the auto successes
passed snapshot || { note snapshot start; rm -rf $RUNS/snapshot_qual
  SE=$(python3 -c "
import glob,json
ok=[]
for f in sorted(glob.glob('$RUNS/manip_qual/results*.jsonl')):
    for l in open(f):
        r=json.loads(l)
        if r['mode']=='auto' and r.get('expert_completed') and (r.get('ovmm') or {}).get('ovmm_place_success')==1.0: ok.append(r['episode_id'])
print(','.join(ok[:4]))")
  GPU=$GPU SPLIT=train EPISODES=$SE OUT=$RUNS/snapshot_qual $RUN $S/snapshot_qual.py > $RUNS/logs/snapshot.log 2>&1
  gate snapshot $RUNS/snapshot_qual; }

# 7. head-camera confirmation in OVMM scenes (frozen neck pitch 0.2 vs neighbours; reported, not a gate)
[ -s $RUNS/camera_sweep/camera_sweep.json ] || { note camera start
  V=""
  for p in 0.1 0.2 0.3; do
    ALPHABOT2_DEPS=$R/deps $RUN -m lmm.embodiments.alphabot2.habitat.build_urdf $R/alphabot2_assets/head_p$p --head-pitch $p >> $RUNS/logs/camera.log 2>&1 || true
    V="$V${V:+,}p$p:$R/alphabot2_assets/head_p$p/alphabot2.urdf"
  done
  GPU=$GPU SPLIT=train HEAD_URDFS=$V EPISODES=$(cat $RUNS/lists/qual_train.txt) OUT=$RUNS/camera_sweep $RUN $S/camera_sweep.py >> $RUNS/logs/camera.log 2>&1
  note camera "done_$( [ -s $RUNS/camera_sweep/camera_sweep.json ] && echo ok || echo failed)"; }

# 8. pilot: 10-20 complete successful trajectories, fresh-reset replay, timing, leakage, both arms, >1 scene
passed pilot || { note pilot start
  rm -rf /dev/shm/alphabot2_claims_pilot
  for w in 0 1; do
    GPU=$GPU WORKER=$((90 + w)) NWORKERS=1 MAX_OK=10 CLAIM_DIR=/dev/shm/alphabot2_claims_pilot \
      EPISODE_FILE=$RUNS/lists/pilot_candidates.txt OUT=$RUNS/collect_v1 $RUN $S/pilot.py > $RUNS/logs/pilot_w$w.log 2>&1 &
  done; wait
  gate pilot $RUNS/collect_v1 $RUNS/lists/pilot_candidates.txt; }

# 9. continuous collection over all feasible TRAIN episodes on every idle GPU (pilot episodes are resumed/skipped)
note collection start
OUT=$RUNS/collect_v1 EPISODE_FILE=$RUNS/lists/collect_train.txt START_GPUS=$GPU PER_GPU=${PER_GPU:-4} \
  CLAIM_DIR=/dev/shm/alphabot2_claims RUN=$S/run_habitat.sh PILOT=$S/pilot.py \
  python3 $S/collect_supervisor.py >> $RUNS/logs/supervisor.log 2>&1
note collection done
