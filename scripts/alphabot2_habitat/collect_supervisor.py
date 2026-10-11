"""Continuous AlphaBot 2 OVMM demonstration collection on whatever GPUs are idle (system python3).

Every POLL_S: a GPU nobody uses (memory.used < IDLE_MIB, utilization < 5 %) is adopted; once adopted
it stays ours unless OUT/collect_control.json lists it in "release_gpus" (the GPU scheduler hands it to
SFT/eval); released GPUs are not re-adopted while listed. On each adopted GPU, PER_GPU collector workers (pilot.py with CLAIM_DIR) are kept alive;
dead workers are restarted (they resume). Other tenants' processes are never touched. Exits when every
episode in EPISODE_FILE has a results row. Writes OUT/supervisor_status.json.

    OUT=... EPISODE_FILE=... START_GPUS=3 PER_GPU=4 python3 collect_supervisor.py
"""
import glob, json, os, re, subprocess, time
from collections import Counter

OUT, EPF = os.environ["OUT"], os.environ["EPISODE_FILE"]
PER_GPU = int(os.environ.get("PER_GPU", "4"))
POLL_S = int(os.environ.get("POLL_S", "300"))
IDLE_MIB = int(os.environ.get("IDLE_MIB", "500"))
CLAIM_DIR = os.environ.get("CLAIM_DIR", "/dev/shm/alphabot2_claims")
RUN = os.environ.get("RUN", "/workspace/lmm/scripts/alphabot2_habitat/run_habitat.sh")
PILOT = os.environ.get("PILOT", "/workspace/lmm/scripts/alphabot2_habitat/pilot.py")
LOGS = os.path.join(os.environ.get("ALPHABOT2_LOG_DIR", "/mnt/cfs/h0ddz1/personal/liyanjun/yanjun/lmm_runtime/logs"), "collect_workers")
os.makedirs(LOGS, exist_ok=True)
eps = open(EPF).read().replace(",", "\n").split()
owned = [int(g) for g in os.environ.get("START_GPUS", "").split(",") if g != ""]
procs = {}                                                     # (gpu, slot) -> Popen
hist = []                                                      # (time, accepted) for the hourly rate


def reason(r):
    if r.get("engineering_error"):
        return "engineering: " + r["engineering_error"].split("(")[0][:60]
    if r.get("success") and not r.get("accepted"):
        return "success_but_replay_mismatch"
    f = r.get("failure") or ""
    f = re.sub(r"\((left|right)\)", "", f).strip()
    return ("expert: " + f[:80]) if f else ("ovmm_not_success" if r.get("expert_completed") else "unknown")



def gpus():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return [tuple(int(x) for x in l.split(",")) for l in q.strip().splitlines()]


def results():
    rs = {}
    # glob + explicit worker files (bosfs readdir lags behind file creation)
    names = set(glob.glob(f"{OUT}/results*.jsonl")) | {f"{OUT}/results_w{i:02d}.jsonl" for i in range(100)}
    for f in sorted(n for n in names if os.path.exists(n)):
        for line in open(f):
            try:
                r = json.loads(line)
                rs[str(r["episode_id"])] = r
            except (ValueError, KeyError):
                pass
    return rs


while True:
    rs = results()
    todo = [e for e in eps if e not in rs]
    alive = {k: p for k, p in procs.items() if p.poll() is None}
    if not todo and not alive:
        break
    # control file (written by gpu_scheduler.py): GPUs to hand back, e.g. for SFT
    try:
        release = set(json.load(open(f"{OUT}/collect_control.json")).get("release_gpus", []))
    except (OSError, ValueError):
        release = set()
    for g in [g for g in owned if g in release]:
        for k in [k for k in procs if k[0] == g]:
            if procs[k].poll() is None:
                os.killpg(procs[k].pid, 15)                       # claims of a dead pid are taken over
            del procs[k]
        owned.remove(g)
        print(time.strftime("%F %T"), "RELEASE_GPU", g, flush=True)
    for idx, mem, util in gpus():
        if idx not in owned and idx not in release and mem < IDLE_MIB and util < 5:
            owned.append(idx)
            print(time.strftime("%F %T"), "ADOPT_GPU", idx, flush=True)
    if todo:
        for g in owned:
            for s in range(PER_GPU):
                k = (g, s)
                if k in alive:
                    continue
                wk = g * 10 + s
                env = {**os.environ, "GPU": str(g), "WORKER": str(wk), "NWORKERS": "1", "CLAIM_DIR": CLAIM_DIR,
                       "EPISODE_FILE": EPF, "OUT": OUT}
                env.pop("MAX_OK", None)
                log = open(f"{LOGS}/w{wk:02d}.log", "a")
                procs[k] = subprocess.Popen(["bash", RUN, PILOT], env=env, stdout=log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True)
                print(time.strftime("%F %T"), "START_WORKER", wk, "gpu", g, "pid", procs[k].pid, flush=True)
    acc = [r for r in rs.values() if r.get("accepted")]
    st = {"time": time.strftime("%F %T"), "episodes": len(eps), "processed": len(rs), "todo": len(todo),
          "accepted": len(acc), "success": sum(bool(r.get("success")) for r in rs.values()),
          "success_but_replay_mismatch": sum(bool(r.get("success")) and not r.get("accepted") for r in rs.values()),
          "engineering_errors": sum("engineering_error" in r for r in rs.values()),
          "accepted_arms": dict(Counter(str(r.get("arm")) for r in acc)),
          "accepted_scenes": len({r.get("scene") for r in acc}),
          "accepted_objects": len({r.get("object") for r in acc}),
          "owned_gpus": owned, "workers_alive": len([p for p in procs.values() if p.poll() is None]),
          "accepted_steps": sum(int(r.get("steps") or 0) for r in acc),
          "accepted_bytes_gb": round(sum(int(r.get("bytes") or 0) for r in acc) / 1e9, 2),
          "accepted_left_fraction": round(sum(r.get("arm") == "left" for r in acc) / max(1, len(acc)), 3),
          "accepted_per_hour_last": None,
          "failure_reasons": dict(Counter(reason(r) for r in rs.values() if not r.get("accepted")).most_common(12)),
          "accepted_by_replica": dict(Counter(int(r.get("replica") or 0) for r in acc))}
    hist.append((time.time(), len(acc)))
    old = [h for h in hist if time.time() - h[0] <= 3600]
    if len(old) >= 2 and old[-1][0] > old[0][0]:
        st["accepted_per_hour_last"] = round((old[-1][1] - old[0][1]) * 3600 / (old[-1][0] - old[0][0]), 1)
    json.dump(st, open(f"{OUT}/supervisor_status.json", "w"), indent=1)
    print("STATUS", json.dumps(st), flush=True)
    time.sleep(POLL_S)
print("###COLLECTION_DONE###", flush=True)
