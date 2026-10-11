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
par() {    # par N LOGNAME ENV... SCRIPT : N parallel shards (WORKER/NWORKERS), wait for all;
           # with GPUS=a,b,c the shards are spread round-robin over those GPUs (overrides GPU=)
  local n=$1 log=$2; shift 2
  local gl=(${GPUS//,/ })
  for w in $(seq 0 $((n - 1))); do
    local args=()
    for x in "$@"; do
      if [ ${#gl[@]} -gt 0 ] && [[ $x == GPU=* ]]; then args+=("GPU=${gl[$((w % ${#gl[@]}))]}"); else args+=("$x"); fi
    done
    env WORKER=$w NWORKERS=$n "${args[@]}" > $RUNS/logs/${log}_w$w.log 2>&1 &
  done; wait
}

# 1. full geometric feasibility re-screen: train 6331 / val 133 / 38 of 38 train scenes
passed screen || { note screen start
  CUDA_VISIBLE_DEVICES=$GPU SPLITS=train,val OUT=$RUNS/feasibility $RUN $S/feasibility_screen.py > $RUNS/logs/screen.log 2>&1
  gate screen $RUNS/feasibility/summary.json $L/artifacts/alphabot2_habitat/feasibility_summary.json; }

# 2. navmesh regression: the documented radius-0.38 val audit (nav_qual defaults: 2 episodes x first 15 scenes),
#    once at the documented settings (r0.38, height 1.7: must reproduce it) and once at the mainline config
passed navmesh || { note navmesh start
  [ -f $RUNS/nav_audit_h17/.done ] || { rm -rf $RUNS/nav_audit_h17
    par 4 nav_audit_h17 GPU=$GPU SPLIT=val NAV_HEIGHT=1.7 ALPHABOT2_NAVMESH_TAG=alphabot2_r038_h17 OUT=$RUNS/nav_audit_h17 $RUN $S/nav_qual.py
    touch $RUNS/nav_audit_h17/.done; }
  [ -f $RUNS/nav_audit/.done ] || { rm -rf $RUNS/nav_audit
    par 4 nav_audit GPU=$GPU SPLIT=val OUT=$RUNS/nav_audit $RUN $S/nav_qual.py
    touch $RUNS/nav_audit/.done; }
  gate navmesh $RUNS/nav_audit_h17 $RUNS/nav_audit; }

# 2b. deterministic embodiment feasibility of every geometry-feasible TRAIN episode (no outcomes):
#     seeded reset reachability per seed-bank replica, pick + place feasibility per arm (navmesh,
#     reach band, aperture along the actual closing axis, IK). Units = (episode, replica).
passed embodiment || { note embodiment start
  par ${EMB_WORKERS:-6} emb SCREEN=$RUNS/feasibility/screen_train.json REPLICAS=${REPLICAS:-3} GPU=$GPU SPLIT=train OUT=$RUNS/emb $RUN $S/emb_screen.py
  gate embodiment $RUNS/emb $RUNS/feasibility/screen_train.json $RUNS/feasibility/units_train.json
  rm -rf $RUNS/lists/*; }

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

# 4. lists from the AlphaBot-feasible TRAIN units only (scene round-robin, object diversity):
#    qualification (replica 0): 5 left-feasible, 5 right-feasible, 8 for auto; 40 pilot candidates;
#    collection: every replica-0 unit (pilot first), then replica 1, then 2, ...
[ -s $RUNS/lists/collect_train.txt ] || python3 - <<PY
import json, random
from collections import defaultdict
U = json.load(open("$RUNS/feasibility/units_train.json"))
def rr(pool, n, seed):
    rnd = random.Random(seed); by = defaultdict(list)
    for u in pool: by[u["scene"]].append(u)
    for v in by.values(): rnd.shuffle(v)
    scenes = sorted(by); rnd.shuffle(scenes); out, used = [], set()
    while len(out) < n and any(by.values()):
        for sc in scenes:
            if by[sc] and len(out) < n:
                pick = next((u for u in by[sc] if u["object"] not in used), by[sc][0])
                by[sc].remove(pick); used.add(pick["object"]); out.append(pick)
    return out
r0 = [u for u in U if u["replica"] == 0]
taken = set()
def take(pool, n, seed):
    out = rr([u for u in pool if u["episode_id"] not in taken], n, seed)
    taken.update(u["episode_id"] for u in out); return out
ql = take([u for u in r0 if "left" in u["feasible_arms"]], 5, 1)
qr = take([u for u in r0 if "right" in u["feasible_arms"]], 5, 2)
qa = take(r0, 8, 3)
pilot = take(r0, 40, 4)
order = [u["unit"] for u in pilot]
for rep in sorted({u["replica"] for u in U}):
    order += [u["unit"] for u in rr([u for u in U if u["replica"] == rep and u["unit"] not in set(order)], 10**9, 10 + rep)]
w = lambda name, xs: open(f"$RUNS/lists/{name}", "w").write(xs)
w("qual_left.txt", ",".join(u["episode_id"] for u in ql)); w("qual_right.txt", ",".join(u["episode_id"] for u in qr))
w("qual_auto.txt", ",".join(u["episode_id"] for u in qa)); w("pilot_candidates.txt", ",".join(u["unit"] for u in pilot))
w("collect_train.txt", "\\n".join(order) + "\\n")
print("LISTS left", len(ql), "right", len(qr), "auto", len(qa), "pilot", len(pilot), "collect_units", len(order))
PY

# 5. manipulation qualification on AlphaBot-feasible TRAIN units: forced-left on left-feasible,
#    forced-right on right-feasible, auto on any feasible; gate: >=1 / >=1 / >=3 complete successes
passed manip || { note manip start; rm -rf $RUNS/manip_qual
  for m in left right; do
    GPU=$GPU SPLIT=train MODES=$m RESULTS_TAG=$m EPISODES=$(cat $RUNS/lists/qual_$m.txt) OUT=$RUNS/manip_qual \
      $RUN $S/manip_qual.py > $RUNS/logs/manip_$m.log 2>&1 &
  done
  par 2 manip_auto GPU=$GPU SPLIT=train MODES=auto EPISODES=$(cat $RUNS/lists/qual_auto.txt) OUT=$RUNS/manip_qual $RUN $S/manip_qual.py
  wait
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
  GPU=$GPU SPLIT=train HEAD_URDFS=$V EPISODES=$(cat $RUNS/lists/qual_auto.txt) OUT=$RUNS/camera_sweep $RUN $S/camera_sweep.py >> $RUNS/logs/camera.log 2>&1
  note camera "done_$( [ -s $RUNS/camera_sweep/camera_sweep.json ] && echo ok || echo failed)"; }

# 7b. FINAL AlphaBot-feasible list: the embodiment screen re-run with the robust place planner
#     (runs/emb_v2; the earlier list is a provisional superset) minus RESET_NONDETERMINISTIC units from the
#     multi-process reset audit (runs/reset_audit). Pilot and collection draw ONLY from this list.
passed embodiment_final || { note embodiment_final start
  while [ -f $RUNS/emb_v2/.running ] || [ -f $RUNS/reset_audit/.running ]; do sleep 60; done   # background runs
  par ${EMB_WORKERS:-6} emb_v2 SCREEN=$RUNS/feasibility/screen_train.json REPLICAS=${REPLICAS:-3} GPU=$GPU SPLIT=train OUT=$RUNS/emb_v2 $RUN $S/emb_screen.py
  python3 $S/reset_audit.py --summarize $RUNS/reset_audit >> $RUNS/logs/reset_audit.log 2>&1
  gate embodiment $RUNS/emb_v2 $RUNS/feasibility/screen_train.json $RUNS/feasibility/units_train_v2.json
  python3 - <<PY
import json
U = json.load(open("$RUNS/feasibility/units_train_v2.json"))
try:
    bad = set(json.load(open("$RUNS/reset_audit/reset_audit.json"))["reset_nondeterministic"])
except (OSError, ValueError):
    bad = set()
bad_eps = {b.split("#")[0] for b in bad}
final = [u for u in U if u["episode_id"] not in bad_eps]
json.dump(final, open("$RUNS/feasibility/units_train_final.json", "w"))
json.dump({"excluded_reset_nondeterministic_episodes": sorted(bad_eps), "units_v2": len(U), "units_final": len(final)},
          open("$RUNS/feasibility/units_train_final_info.json", "w"), indent=1)
print("FINAL_UNITS", len(U), "->", len(final), "excluded", sorted(bad_eps))
PY
  cp $RUNS/feasibility/units_train_final.json $RUNS/feasibility/units_train.json      # data gates read this name
  rm -f $RUNS/lists/pilot_candidates.txt $RUNS/lists/collect_train.txt
  python3 - <<PY
import json, random
from collections import defaultdict
U = json.load(open("$RUNS/feasibility/units_train_final.json"))
used = set()
for f in ("qual_left", "qual_right", "qual_auto"):
    try:
        used |= set(open(f"$RUNS/lists/{f}.txt").read().strip().split(","))
    except OSError:
        pass
def rr(pool, n, seed):
    rnd = random.Random(seed); by = defaultdict(list)
    for u in pool: by[u["scene"]].append(u)
    for v in by.values(): rnd.shuffle(v)
    scenes = sorted(by); rnd.shuffle(scenes); out, seen = [], set()
    while len(out) < n and any(by.values()):
        for sc in scenes:
            if by[sc] and len(out) < n:
                pick = next((u for u in by[sc] if u["object"] not in seen), by[sc][0])
                by[sc].remove(pick); seen.add(pick["object"]); out.append(pick)
    return out
pilot = rr([u for u in U if u["replica"] == 0 and u["episode_id"] not in used], 40, 4)
order = [u["unit"] for u in pilot]
for rep in sorted({u["replica"] for u in U}):
    order += [u["unit"] for u in rr([u for u in U if u["replica"] == rep and u["unit"] not in set(order)], 10**9, 10 + rep)]
open("$RUNS/lists/pilot_candidates.txt", "w").write(",".join(u["unit"] for u in pilot))
open("$RUNS/lists/collect_train.txt", "w").write("\\n".join(order) + "\\n")
print("FINAL_LISTS pilot", len(pilot), "collect_units", len(order))
PY
  note embodiment_final lists_frozen
  echo '{"gate": "embodiment_final", "pass": true}' >> $ST; }

# 8. pilot: 10-20 complete successful trajectories, fresh-reset replay, timing, leakage, both arms, >1 scene
passed pilot || { note pilot start
  rm -rf /dev/shm/alphabot2_claims_pilot
  for w in 0 1; do
    GPU=$GPU WORKER=$((90 + w)) NWORKERS=1 MAX_OK=10 CLAIM_DIR=/dev/shm/alphabot2_claims_pilot \
      EPISODE_FILE=$RUNS/lists/pilot_candidates.txt OUT=$RUNS/collect_v1 $RUN $S/pilot.py > $RUNS/logs/pilot_w$w.log 2>&1 &
  done; wait
  gate pilot $RUNS/collect_v1 $RUNS/lists/pilot_candidates.txt; }

# 9. long-running: continuous collection (seed-bank units, idle GPUs, crash recovery, atomic claims,
#    resume) -> data QC gate -> conversion -> pi0.5 SFT -> normal-init e2e eval (gpu_scheduler.py)
note collection start
python3 $S/gpu_scheduler.py ${SCHEDULE:-$L/configs/alphabot2_pi05/schedule.json} >> $RUNS/logs/scheduler.log 2>&1
note collection exited
