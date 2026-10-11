#!/bin/bash
# Background (tmux) runs feeding the pipeline's final list stage: the robust-planner embodiment screen
# (runs/emb_v2) and the multi-process A->B->A reset audit (runs/reset_audit). Each keeps a .running
# marker until done; pipeline stage 7b waits for both.
#     EMB_GPUS=2,3 AUDIT_GPU=3 bash start_final_screen.sh
set -u
L=/workspace/lmm; S=$L/scripts/alphabot2_habitat; R=/workspace/runtime/runs; RUN="bash $S/run_habitat.sh"
mkdir -p $R/emb_v2 $R/reset_audit $R/logs
touch $R/emb_v2/.running $R/reset_audit/.running
GL=(${EMB_GPUS//,/ }); N=${EMB_WORKERS:-16}
(
  for w in $(seq 0 $((N - 1))); do
    env WORKER=$w NWORKERS=$N SCREEN=$R/feasibility/screen_train.json REPLICAS=3 GPU=${GL[$((w % ${#GL[@]}))]} SPLIT=train \
      OUT=$R/emb_v2 $RUN $S/emb_screen.py > $R/logs/emb_v2_w$w.log 2>&1 &
  done; wait; rm -f $R/emb_v2/.running; echo EMB_V2_DONE
) &
(
  python3 - > $R/reset_audit/plan.json <<PY
import json, random
U = [u for u in json.load(open("$R/feasibility/units_train.json")) if u["replica"] == 0]
rnd = random.Random(7)
pick = [u["unit"] for u in rnd.sample(U, 36)]
qual = []
for f in ("qual_left", "qual_right", "qual_auto"):
    qual += [e + "#0" for e in open(f"$R/lists/{f}.txt").read().strip().split(",") if e]
units = sorted(set(pick + qual + ["31991#0"]))
pairs = ["31991#0:1798#0", "31991#0:31226#0"] + [f"{a}:{b}" for a, b in zip(pick[:10], pick[10:20])]
print(json.dumps({"units": units, "pairs": pairs}))
PY
  U=$(python3 -c "import json;print(','.join(json.load(open('$R/reset_audit/plan.json'))['units']))")
  P=$(python3 -c "import json;print(','.join(json.load(open('$R/reset_audit/plan.json'))['pairs']))")
  for w in 0 1 2 3; do
    env WORKER=$w NWORKERS=4 UNITS=$U PAIRS=$P GPU=${AUDIT_GPU:-3} SPLIT=train OUT=$R/reset_audit $RUN $S/reset_audit.py \
      > $R/logs/reset_audit_w$w.log 2>&1 &
  done; wait
  python3 $S/reset_audit.py --summarize $R/reset_audit; rm -f $R/reset_audit/.running; echo RESET_AUDIT_DONE
) &
wait
