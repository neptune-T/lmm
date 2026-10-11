#!/bin/bash
# One-screen status of the AlphaBot 2 OVMM long run. tmux windows (session lmm) run it in a loop:
#   status.sh collect   collection supervisor (rates, failure reasons, GPUs)
#   status.sh jobs      scheduler state, blocks, current job (conversion / interface gate / SFT / eval) log tail
#   status.sh           both
R=${RUNS:-/workspace/runtime/runs}
LOGS=${LOGS:-/mnt/cfs/h0ddz1/personal/liyanjun/yanjun/lmm_runtime/logs}
WHAT=${1:-all}
echo "=== $(date '+%F %T')  [$WHAT]   attach: TMUX_TMPDIR=/dev/shm /mnt/cfs/h0ddz1/personal/liyanjun/yanjun/envs/tools/bin/tmux attach -t lmm"
WHAT=$WHAT R=$R LOGS=$LOGS python3 - <<'PY' 2>&1
import json, os, subprocess
R, LOGS, WHAT = os.environ["R"], os.environ["LOGS"], os.environ["WHAT"]


def tail(f, n):
    try:
        lines = [l for l in open(f, errors="replace").read().splitlines() if "Svt[" not in l and "SVT [" not in l and "mp4 @" not in l]
        return "\n".join("    " + l[:200] for l in lines[-n:])
    except OSError:
        return "    (no log)"


def procs(pat):
    return subprocess.run(["pgrep", "-af", pat], capture_output=True, text=True).stdout.count("\n")


if WHAT in ("collect", "all"):
    try:
        s = json.load(open(f"{R}/collect_clean_v1/supervisor_status.json"))
        succ = s.get("success") or 0
        print("--- collection (collect_clean_v1)")
        print("   ", json.dumps({k: s.get(k) for k in ("time", "processed", "todo", "accepted", "success", "success_but_replay_mismatch",
                                                     "accepted_per_hour_last", "accepted_left_fraction", "accepted_scenes",
                                                     "owned_gpus", "workers_alive")}))
        print(f"    replay reject fraction of successes: {s.get('success_but_replay_mismatch', 0) / max(1, succ):.3f}")
        print("    failure reasons:", json.dumps(s.get("failure_reasons")))
    except Exception as e:
        print("--- collection: no status", e)
    print("    supervisor procs:", procs("collect_supervisor.py"), " pilot workers:", procs("alphabot2_habitat/pilot.py"))
if WHAT in ("jobs", "all"):
    try:
        st = json.load(open(f"{R}/scheduler_state.json"))
        print("--- scheduler: procs", procs("alphabot2_habitat/gpu_scheduler.py"), " watchdog", procs("scheduler_watchdog.sh"))
        print("    rounds:", json.dumps({k: v.get("status") for k, v in st["rounds"].items()}), " h_exec_frozen:", st.get("h_exec_frozen"))
        print("    BLOCKED:", json.dumps(st.get("blocked")) if st.get("blocked") else "no")
        if st.get("stage_complete"):
            print("    STAGE COMPLETE:", json.dumps(st["stage_complete"]))
        for j in st["jobs"]:
            print(f"    {j['round']:10s} {j['kind']:8s} {str(j.get('variant')):9s} {j['status']:9s}",
                  {k: j[k] for k in ("pass", "attempts", "gpus") if k in j}, "done_h=" + ",".join(j.get("done_h", {})) if j["kind"] == "hsweep" else "")
        print("    val screen procs:", procs("val_screen.sh"), " units_val:", os.path.exists(f"{R}/feasibility/units_val.json"))
        run = next((j for j in st["jobs"] if j["status"] == "running"), None) or next((j for j in st["jobs"] if j["status"] == "pending"), None)
        if run:
            rnd, k, v = run["round"], run["kind"], run.get("variant")
            log = {"convert": f"{LOGS}/convert_{rnd}_train.log", "iface": f"{LOGS}/interface_gate_{rnd}.log"}.get(k)
            if k == "convert" and os.path.exists(f"{LOGS}/convert_{rnd}_val.log"):
                log = f"{LOGS}/convert_{rnd}_val.log"
            if k == "sft":
                log = f"{LOGS}/sft_{rnd}_{v}_attempt{run.get('attempts', 0)}.log"
            if k in ("hsweep", "eval"):
                d = f"{R}/eval/{rnd}_{v}"
                sub = sorted((x for x in os.listdir(d)), key=lambda x: os.path.getmtime(f"{d}/{x}")) if os.path.isdir(d) else []
                log = f"{d}/{sub[-1]}/worker0.log" if sub else None
            print(f"--- current job: {rnd} {k} {v}   log: {log}")
            print(tail(log, 8) if log else "    (none)")
    except Exception as e:
        print("--- scheduler: no state", e)
    print("--- last scheduler events")
    print(tail(f"{R}/scheduler_events.jsonl", 6))
PY
echo "--- GPUs (index, used MiB, util %)"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
