#!/bin/bash
# One-screen status of the AlphaBot 2 OVMM long run (tmux "status" window: watch-like loop).
R=${RUNS:-/workspace/runtime/runs}
echo "=== $(date '+%F %T')   (tmux attach -t lmm; windows: pipeline / status / logs)"
echo "--- last pipeline gates/events"; tail -n 6 $R/pipeline_status.jsonl 2>/dev/null | cut -c1-230
python3 - <<PY 2>/dev/null
import json, os
R = "$R"
ids = set()
for i in range(64):
    f = f"{R}/emb/emb_w{i:02d}.jsonl"
    if os.path.exists(f):
        ids |= {json.loads(l)["episode_id"] for l in open(f) if l.strip()}
print(f"--- embodiment screen: {len(ids)} / 6331 geometry-feasible TRAIN episodes screened")
try:
    s = json.load(open(f"{R}/collect_v1/supervisor_status.json"))
    print("--- collection:", json.dumps({k: s.get(k) for k in ("time", "processed", "todo", "accepted", "accepted_left_fraction",
          "accepted_scenes", "accepted_objects", "accepted_bytes_gb", "accepted_per_hour_last", "success_but_replay_mismatch",
          "engineering_errors", "owned_gpus", "workers_alive")}))
    print("    failure reasons:", json.dumps(s.get("failure_reasons")))
except Exception:
    print("--- collection: not started")
try:
    st = json.load(open(f"{R}/scheduler_state.json"))
    print("--- scheduler rounds:", json.dumps({k: v.get("status") for k, v in st["rounds"].items()}), "blocked:", st.get("blocked"))
    print("    jobs:", [(j["round"], j["kind"], j.get("variant"), j["status"]) for j in st["jobs"]])
except Exception:
    print("--- scheduler: not started")
PY
echo "--- GPUs"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "--- pi0.5 env build"; tail -n 1 /workspace/runtime/logs/build_pi05_env.log 2>/dev/null | cut -c1-160
