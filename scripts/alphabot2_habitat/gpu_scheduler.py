"""Long-running AlphaBot 2 OVMM scheduler (system python3, stdlib only).

Collection runs continuously. Tiered pi0.5 SFT rounds run IN PARALLEL with it (schedule.json "rounds"):
each round starts when its data gate passes on the accepted data at that moment (QC over every accepted
trajectory, no replay mismatch anywhere, enough trajectories / scenes / both arms), converts a snapshot,
trains the balanced variant, evaluates it end-to-end from normal OVMM initialization, then (lower
priority) the uniform baseline if listed. The next round is created only after the previous round's
balanced eval shows stable non-zero full success; otherwise rounds halt and a funnel diagnosis is
written (which of find / grasp / carry / place drops), so data is not piled up blindly.

* GPUs: usable only with no foreign compute process (pid not visible in this container). SFT takes idle
  GPUs first, then collection GPUs beyond min_collect_gpus (handed back via collect_control.json).
  Other users' processes are never touched or preempted. One GPU job (SFT or eval) at a time.
* Hard blocks stop SFT/eval work (collection continues) and are recorded with their reason: data QC
  integrity/determinism failure, conversion failure, SFT non-finite loss (exit 3) or restarts exhausted,
  val feasibility screen incomplete, policy server failure, eval failing on every episode.
  A missing SFT env/assets is a wait (re-checked every poll), not a block.
* SFT crash/stall -> restart with resume (newest complete checkpoint), up to max_restarts.
* RL / frontier are never started. State: RUNS/scheduler_state.json; events: RUNS/scheduler_events.jsonl.

    python3 gpu_scheduler.py /workspace/lmm/configs/alphabot2_pi05/schedule.json
"""
import json, os, signal, subprocess, sys, time

L = os.environ.get("LMM_REPO", "/workspace/lmm")
S = f"{L}/scripts/alphabot2_habitat"
RUN = ["bash", f"{S}/run_habitat.sh"]
CFG = json.load(open(sys.argv[1]))
R = CFG["runs"]
STATE_F, EV_F = f"{R}/scheduler_state.json", f"{R}/scheduler_events.jsonl"
os.makedirs(f"{R}/logs", exist_ok=True)
st = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {}
st.setdefault("rounds", {r["name"]: {"status": "waiting"} for r in CFG["rounds"]})
st.setdefault("jobs", [])
st.setdefault("halt_rounds", False)
st.setdefault("blocked", None)
procs = {}
ROUNDS = {r["name"]: r for r in CFG["rounds"]}
ORDER = [r["name"] for r in CFG["rounds"]]
KIND_PRIO = {"convert": 0, "iface": 1, "sft": 2, "hsweep": 3, "eval": 4}


def save():
    json.dump(st, open(STATE_F + ".tmp", "w"), indent=1)
    os.replace(STATE_F + ".tmp", STATE_F)


def event(kind, **kw):
    rec = {"time": time.strftime("%F %T"), "event": kind, **kw}
    with open(EV_F, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(json.dumps(rec, default=str), flush=True)


def block(gate_name, reason, **kw):
    st["blocked"] = {"gate": gate_name, "reason": reason, "time": time.strftime("%F %T"), **kw}
    save()
    event("BLOCKED", gate=gate_name, reason=reason, **kw)


def sh(cmd, log, env=None, cwd=None):
    return subprocess.Popen(cmd, env={**os.environ, **(env or {})}, cwd=cwd, stdout=open(log, "a"),
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


def gate(*args):
    out = subprocess.run(RUN + [f"{S}/gates.py", *args], capture_output=True, text=True).stdout
    line = next((l for l in out.splitlines() if l.startswith("###GATE###")), None)
    res = json.loads(line[10:]) if line else {"pass": False, "error": "gate produced no output"}
    with open(f"{R}/pipeline_status.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    return res


def gpus():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout
    foreign = {u.strip() for p, u in (l.split(",") for l in apps.strip().splitlines() if l.strip())
               if not os.path.exists(f"/proc/{p.strip()}")}
    out = {}
    for l in q.strip().splitlines():
        i, uuid, used, tot, util = [x.strip() for x in l.split(",")]
        out[int(i)] = {"index": int(i), "used": int(used), "free": int(tot) - int(used), "util": int(util),
                       "foreign": uuid in foreign}
    return out


def control(release):
    c = f"{CFG['collect']['out']}/collect_control.json"
    json.dump({"release_gpus": sorted(release)}, open(c + ".tmp", "w"))
    os.replace(c + ".tmp", c)


def collect_status():
    try:
        return json.load(open(f"{CFG['collect']['out']}/supervisor_status.json"))
    except (OSError, ValueError):
        return {}


def ensure_collection():
    p = procs.get("collect")
    if p is not None and p.poll() is None:
        return
    if p is not None:
        event("collection_supervisor_restart", returncode=p.returncode)
    C = CFG["collect"]
    procs["collect"] = sh(["python3", f"{S}/collect_supervisor.py"], f"{R}/logs/supervisor.log",
                          env={"OUT": C["out"], "EPISODE_FILE": C["units_file"], "START_GPUS": ",".join(map(str, C["start_gpus"])),
                               "PER_GPU": str(C["per_gpu"]), "POLL_S": str(C["poll_s"]), "CLAIM_DIR": "/dev/shm/alphabot2_claims",
                               "RUN": f"{S}/run_habitat.sh", "PILOT": f"{S}/pilot.py"})


def acquire(n_min, n_max, min_free):
    """Up to n_max (at least n_min) GPUs with no foreign process: idle ones first, then collection GPUs
    beyond min_collect_gpus, which collection hands back. Returns the list or None."""
    owned = set(collect_status().get("owned_gpus", []))
    g = gpus()
    ok = [i for i, x in g.items() if not x["foreign"]]
    idle = [i for i in ok if i not in owned and g[i]["used"] < 500 and g[i]["free"] >= min_free]
    ours = [i for i in ok if i in owned]
    spare = ours[CFG["collect"]["min_collect_gpus"]:]
    chosen = (idle + spare)[:n_max]
    if len(chosen) < n_min:
        return None
    if any(i in owned for i in chosen):
        control(chosen)
        time.sleep(CFG["collect"]["poll_s"] + 30)                 # the supervisor stops its workers there
    g = gpus()
    chosen = [i for i in chosen if not g[i]["foreign"] and g[i]["free"] >= min_free]
    if len(chosen) < n_min:
        control([])
        event("gpus_not_free", wanted=n_min)
        return None
    control(chosen)
    return chosen


def env_ready():
    A, py = CFG["sft"]["assets"], CFG["sft"]["python"]
    return [p for p in (py, f"{A}/ckpt/config.json", f"{A}/paligemma_tokenizer.model") if not os.path.exists(p)]


def env_pi():
    A = CFG["sft"]["assets"]
    return {"PI05_CKPT": f"{A}/ckpt", "PI05_SPM": f"{A}/paligemma_tokenizer.model", "TORCHDYNAMO_DISABLE": "1",
            "LMM_REPO": L, "PYTHONUNBUFFERED": "1",
            "HF_HOME": os.path.join(os.path.dirname(CFG["sft"]["python"]), "..", "..", "_hf_home")}


def ds_dir(rnd, split):
    return f"{CFG['sft']['datasets']}/alphabot2_ovmm_{rnd}_{split}"


def sft_config(rnd, variant):
    """Round/variant config = the variant's base yaml with name/out/data/steps replaced."""
    name = f"{rnd}_{variant}"
    out_cfg = f"{R}/sft_configs/{name}.yaml"
    os.makedirs(os.path.dirname(out_cfg), exist_ok=True)
    rep = {"name": name, "out": f"{CFG['sft']['out']}/{name}", "train_data": ds_dir(rnd, "train"),
           "val_data": ds_dir(rnd, "val"), "steps": str(ROUNDS[rnd]["steps"]), "init_checkpoint": f"{CFG['sft']['assets']}/ckpt"}
    lines = []
    for l in open(CFG["sft"]["configs"][variant]):
        k = l.split(":", 1)[0]
        lines.append(f"{k}: {rep[k]}\n" if (":" in l and not l.startswith((" ", "#")) and k in rep) else l)
    open(out_cfg, "w").writelines(lines)
    return out_cfg, rep["out"], name


# ------------------------------------------------------------------ data gate -> round jobs
def check_data():
    if st["blocked"] or st["halt_rounds"] or time.time() - st.get("last_data_check", 0) < CFG["data_check_every_s"]:
        return
    nxt = next((n for n in ORDER if st["rounds"][n]["status"] == "waiting"), None)
    if nxt is None:
        return
    i = ORDER.index(nxt)
    if i > 0 and st["rounds"][ORDER[i - 1]]["status"] != "decided_ok":
        return                                                     # previous round not yet shown to work
    st["last_data_check"] = time.time()
    r = ROUNDS[nxt]
    res = gate("dataqc", CFG["collect"]["out"], f"{R}/feasibility/units_train.json", str(r["min_accepted"]),
               str(r["min_arm_share"]), str(r["min_scene_coverage"]), str(r["min_scenes"]))
    keep = {k: res.get(k) for k in ("pass", "accepted", "scenes", "arm_share", "scene_coverage",
                                    "n_integrity_failures", "success_but_replay_mismatch")}
    st["rounds"][nxt]["last_check"] = keep
    save()
    event("data_check", round=nxt, **keep)
    if res.get("n_integrity_failures") or res.get("success_but_replay_mismatch"):
        block("DATAQC", "integrity/timing/leakage failure or determinism break in accepted data", round=nxt, result=res)
        return
    if res.get("pass"):
        st["rounds"][nxt].update({"status": "running", "snapshot_accepted": res["accepted"]})
        st["jobs"].append({"round": nxt, "kind": "convert", "variant": None, "status": "pending"})
        st["jobs"].append({"round": nxt, "kind": "iface", "variant": None, "status": "pending"})
        for v in r["variants"]:
            st["jobs"].append({"round": nxt, "kind": "sft", "variant": v, "status": "pending", "attempts": 0})
            if v == "balanced" and st.get("h_exec_frozen") is None and not any(x["kind"] == "hsweep" for x in st["jobs"]):
                st["jobs"].append({"round": nxt, "kind": "hsweep", "variant": v, "status": "pending", "done_h": {}})
            st["jobs"].append({"round": nxt, "kind": "eval", "variant": v, "status": "pending", "pass": 0})
        save()
        event("round_start", round=nxt, accepted=res["accepted"])


def prio(j):
    # every balanced job (and conversions) before any uniform baseline job; then round order, then kind
    return (0 if j["variant"] in (None, "balanced") else 1, ORDER.index(j["round"]), KIND_PRIO[j["kind"]])


# ------------------------------------------------------------------ jobs
def run_convert(j):
    r = ROUNDS[j["round"]]
    for split in ("train", "val"):
        cmd = [CFG["sft"]["python"], f"{L}/scripts/alphabot2_pi05/ovmm_to_lerobot.py", "--src", CFG["collect"]["out"],
               "--split", split, "--out", ds_dir(j["round"], split)]
        if split == "train" and r["max_episodes"]:
            cmd += ["--max-episodes", str(r["max_episodes"])]
        rc = subprocess.run(cmd, env={**os.environ, **env_pi()}, stdout=open(f"{R}/logs/convert_{j['round']}_{split}.log", "a"),
                            stderr=subprocess.STDOUT).returncode
        if rc != 0:
            j["status"] = "blocked"
            block("CONVERT", f"ovmm_to_lerobot.py {j['round']} --split {split} exited {rc}")
            return
    j["status"] = "done"
    event("converted", round=j["round"])


def run_iface(j):
    """Policy-interface gate on the round's converted dataset (hard gate before any SFT of the round)."""
    cfg, _, _ = sft_config(j["round"], "balanced")
    g = acquire(1, 1, CFG["eval"]["server_min_free_mib"])
    if g is None:
        return
    out = f"{R}/interface_gate/{j['round']}"
    rc = subprocess.run([CFG["sft"]["python"], f"{L}/scripts/alphabot2_pi05/interface_gate.py", "--config", cfg,
                         "--collect", CFG["collect"]["out"], "--h-exec", str(h_exec()), "--out", out],
                        env={**os.environ, **env_pi(), "CUDA_VISIBLE_DEVICES": str(g[0])}, cwd=f"{L}/scripts/alphabot2_pi05",
                        stdout=open(f"{R}/logs/interface_gate_{j['round']}.log", "a"), stderr=subprocess.STDOUT).returncode
    control([])
    try:
        res = json.load(open(f"{out}/interface_gate.json"))
    except (OSError, ValueError):
        res = {"pass": False, "error": f"interface_gate.py exited {rc} without a report"}
    with open(f"{R}/pipeline_status.jsonl", "a") as f:
        f.write(json.dumps({"gate": "interface", "round": j["round"], "pass": res.get("pass"), "checks": res.get("checks")}) + "\n")
    if rc == 0 and res.get("pass"):
        j["status"] = "done"
        event("interface_gate_pass", round=j["round"], checks=res.get("checks"), visualization=res.get("report", {}).get("visualization"))
    else:
        j["status"] = "blocked"
        block("INTERFACE", "policy-interface gate failed", round=j["round"],
              failed=[k for k, v in (res.get("checks") or {}).items() if not v], error=res.get("error"))


def run_sft(j):
    cfg, out, name = sft_config(j["round"], j["variant"])
    p = procs.get("sft")
    if p is None:                                                   # (re)launch, resuming if checkpoints exist
        if j.get("attempts", 0) > CFG["sft"]["max_restarts"]:
            j["status"] = "blocked"
            block("SFT", "restarts exhausted", run=name)
            return
        chosen = acquire(CFG["sft"]["min_gpus"], CFG["sft"]["max_gpus"], CFG["sft"]["min_free_mib"])
        if chosen is None:
            return
        procs["sft"] = sh([CFG["sft"]["python"], "-m", "torch.distributed.run", f"--nproc_per_node={len(chosen)}",
                           f"--master_port={CFG['sft']['master_port']}", f"{L}/scripts/alphabot2_pi05/sft_ovmm.py",
                           "--config", cfg], f"{R}/logs/sft_{name}_attempt{j['attempts']}.log",
                          env={**env_pi(), "CUDA_VISIBLE_DEVICES": ",".join(map(str, chosen))}, cwd=f"{L}/scripts/alphabot2_pi05")
        j.update({"status": "running", "gpus": chosen})
        save()
        event("sft_launch", run=name, gpus=chosen, attempt=j["attempts"])
        return
    if p.poll() is None:
        try:
            hb = json.load(open(f"{out}/heartbeat.json"))
            if time.time() - hb["time"] > CFG["sft"]["stall_s"]:
                event("sft_stall_kill", run=name, heartbeat=hb)
                os.killpg(p.pid, signal.SIGTERM)
        except (OSError, ValueError):
            pass
        return
    rc = p.returncode
    del procs["sft"]
    control([])
    if rc == 0 and os.path.exists(f"{out}/DONE"):
        j["status"] = "done"
        event("sft_done", run=name, summary=json.load(open(f"{out}/summary.json")))
    elif rc == 3:
        j["status"] = "blocked"
        block("SFT", "non-finite loss (exit 3)", run=name)
    else:
        j["attempts"] = j.get("attempts", 0) + 1
        event("sft_exit_restart", run=name, returncode=rc, attempt=j["attempts"])


def val_units():
    f = f"{R}/feasibility/units_{CFG['eval']['split']}.json"
    if os.path.exists(f):
        return f
    E = CFG["eval"]
    g = acquire(1, 1, E["server_min_free_mib"])
    if g is None:
        return None
    subprocess.run(RUN + [f"{S}/emb_screen.py"], env={**os.environ, "SCREEN": f"{R}/feasibility/screen_{E['split']}.json",
                                                     "SPLIT": E["split"], "REPLICAS": str(E["replicas"]), "GPU": str(g[0]),
                                                     "OUT": f"{R}/emb_{E['split']}"},
                   stdout=open(f"{R}/logs/emb_{E['split']}.log", "a"), stderr=subprocess.STDOUT)
    control([])
    res = gate("embodiment", f"{R}/emb_{E['split']}", f"{R}/feasibility/screen_{E['split']}.json", f)
    if not res.get("pass"):
        if os.path.exists(f):
            os.remove(f)
        block("EVAL_SCREEN", "val embodiment screen incomplete/non-deterministic", result=res)
        return None
    st["val_funnel"] = res.get("funnel_episodes")
    event("val_screen_ok", funnel=res.get("funnel_episodes"), units=res.get("units"))
    return f


def h_exec():
    """The ONE execution horizon: frozen after the sanity sweep; before that the preferred value."""
    return int(st.get("h_exec_frozen") or CFG["eval"]["h_exec_preferred"])


def eval_pass(name, sft_out, units, h, seed_offset, out):
    """One end-to-end eval pass (policy server + Habitat workers on one free GPU). Returns the summary,
    None to retry later (no GPU), or {"blocked": reason}."""
    E = CFG["eval"]
    g = acquire(1, 1, E["server_min_free_mib"])
    if g is None:
        return None
    gpu = g[0]
    os.makedirs(out, exist_ok=True)
    server = sh([CFG["sft"]["python"], f"{L}/scripts/alphabot2_pi05/policy_server.py", "--ckpt", f"{sft_out}/best",
                 "--port", str(E["port"])], f"{out}/server.log", env={**env_pi(), "CUDA_VISIBLE_DEVICES": str(gpu)},
                cwd=f"{L}/scripts/alphabot2_pi05")
    t0 = time.time()
    while time.time() - t0 < 1800 and server.poll() is None and "READY" not in open(f"{out}/server.log").read():
        time.sleep(10)
    if server.poll() is not None or "READY" not in open(f"{out}/server.log").read():
        control([])
        return {"blocked": "policy server failed to start", "log": open(f"{out}/server.log").read()[-1500:]}
    ws = [sh(RUN + [f"{S}/eval_policy.py"], f"{out}/worker{w}.log",
             env={"EPISODES": ",".join(units), "SPLIT": E["split"], "PORT": str(E["port"]), "OUT": out, "GPU": str(gpu),
                  "WORKER": str(w), "NWORKERS": str(E["workers"]), "EXEC_STEPS": str(h), "SEED_OFFSET": str(seed_offset)})
          for w in range(E["workers"])]
    for w in ws:
        w.wait()
    os.killpg(server.pid, signal.SIGTERM)
    control([])
    s = json.loads(subprocess.run(["python3", f"{S}/eval_summary.py", out], capture_output=True, text=True)
                   .stdout.split("###EVAL_SUMMARY###", 1)[1])
    s.update({"h_exec": h, "h_pred": CFG["eval"]["h_pred"], "seed_offset": seed_offset})
    if s["engineering_errors"] == s["episodes"]:
        return {"blocked": "every eval episode failed with an engineering error", "summary": s}
    return s


def run_hsweep(j):
    """Choose the ONE execution horizon on a small fixed subset after the first balanced SFT, then freeze it."""
    E = CFG["eval"]
    _, sft_out, name = sft_config(j["round"], j["variant"])
    uf = val_units()
    if uf is None:
        return
    units = [u["unit"] for u in json.load(open(uf))][: E["sweep_units"]]
    j["status"] = "running"
    for h in E["h_exec_sweep"]:
        if str(h) in j["done_h"]:
            continue
        s = eval_pass(name, sft_out, units, h, 0, f"{R}/eval/{name}/hsweep_h{h}")
        if s is None:
            return
        if "blocked" in s:
            j["status"] = "blocked"
            block("HSWEEP", s["blocked"], run=name, h_exec=h)
            return
        j["done_h"][str(h)] = s
        save()
        event("hsweep_point", run=name, h_exec=h, full=f"{s['full_success_count']}/{s['overall']['n']}", overall=s["overall"])
    stages = ("full", "place", "carry", "grasp", "find")
    counts = {int(h): {k: int(round((s["overall"][k] or 0) * s["overall"]["n"])) for k in stages} for h, s in j["done_h"].items()}
    n = max(s["overall"]["n"] for s in j["done_h"].values())
    key = next((k for k in stages if max(c[k] for c in counts.values()) > 0), "find")
    best = max(c[key] for c in counts.values())
    tol = max(1, -(-n // 10))                                       # max(1, ceil(10% of n))
    pref = E["h_exec_preferred"]
    if pref in counts and counts[pref][key] >= best - tol:
        chosen = pref
    else:
        chosen = min((h for h in counts if counts[h][key] == best), key=lambda h: abs(h - pref))
    st["h_exec_frozen"] = int(chosen)
    j["status"] = "done"
    rec = {"chosen_h_exec": chosen, "preferred": pref, "compared_on_stage": key, "tolerance": tol, "units": n,
           "counts_by_h": counts, "h_pred": E["h_pred"], "note": "frozen for every later SFT/eval; not changed per round"}
    json.dump(rec, open(f"{R}/eval/{name}/hsweep_decision.json", "w"), indent=1)
    save()
    event("h_exec_frozen", **rec)


def run_eval(j):
    E = CFG["eval"]
    _, sft_out, name = sft_config(j["round"], j["variant"])
    uf = val_units()
    if uf is None or st.get("h_exec_frozen") is None:
        return                                                     # eval only with the frozen protocol
    units = [u["unit"] for u in json.load(open(uf))]
    while j["pass"] < E["passes"]:
        j["status"] = "running"
        save()
        s = eval_pass(name, sft_out, units, h_exec(), j["pass"], f"{R}/eval/{name}/pass{j['pass']}")
        if s is None:
            return
        if "blocked" in s:
            j["status"] = "blocked"
            block("EVAL", s["blocked"], run=name)
            return
        j.setdefault("summaries", []).append(s)
        j["pass"] += 1
        save()
        event("eval_pass_done", run=name, n=j["pass"], h_exec=s["h_exec"], success=f"{s['full_success_count']}/{s['overall']['n']}",
              overall=s["overall"], by_hand=s["by_hand"])
    j["status"] = "done"
    decide(j, name)


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def decide(j, name):
    ss = j["summaries"]
    stages = ("find", "grasp", "carry", "place", "full")
    k_min = CFG["eval"]["min_full_success_per_pass"]
    passes = []
    for s in ss:
        n = s["overall"]["n"]
        counts = {k: int(round((s["overall"][k] or 0) * n)) for k in stages}
        passes.append({"eval_count": n, "success_count": s["full_success_count"],
                       "success_rate": round(s["full_success_count"] / max(1, n), 4),
                       "success_rate_wilson95": wilson(s["full_success_count"], n),
                       "funnel_counts": counts, "engineering_errors": s["engineering_errors"],
                       "by_hand": s["by_hand"], "by_scene": s["by_scene"]})
    works = all(p["success_count"] >= k_min for p in passes)
    drop = None
    if not works:                                                  # where does the funnel collapse?
        m = {k: sum(p["funnel_counts"][k] for p in passes) / max(1, sum(p["eval_count"] for p in passes)) for k in stages}
        prev = 1.0
        for k in stages:
            if m[k] < 0.5 * prev or m[k] < 0.02:
                drop = {"stage": k, "rate": round(m[k], 4), "previous": round(prev, 4)}
                break
            prev = m[k]
    sft_sum = {}
    try:
        sft_sum = json.load(open(f"{sft_config(j['round'], j['variant'])[1]}/summary.json"))
    except (OSError, ValueError):
        pass
    dec = {"run": name, "round": j["round"], "variant": j["variant"], "h_exec": h_exec(), "h_exec_frozen": st.get("h_exec_frozen"),
           "h_pred": CFG["eval"]["h_pred"],
           "snapshot_accepted": st["rounds"][j["round"]].get("snapshot_accepted"),
           "sft": {k: sft_sum.get(k) for k in ("steps", "samples_seen", "effective_epochs", "train_episodes", "train_frames",
                                               "global_batch", "best_step", "best_val_loss")},
           "eval_split": CFG["eval"]["split"], "val_funnel_full_to_feasible": st.get("val_funnel"),
           "passes": passes, "criterion": f">= {k_min} full successes in every noise-seed pass",
           "works": works, "first_collapsing_stage": drop,
           "next": ("continue rounds; frontier/FP8/RL wait for the user's review" if works else
                    "rounds halted: inspect the collapsing funnel stage and SFT/data coverage; do not add data blindly"),
           "note": "AlphaBot-feasible val subset from normal OVMM initialization; not standard OVMM; RL never auto-started"}
    os.makedirs(f"{R}/eval/{name}", exist_ok=True)
    json.dump(dec, open(f"{R}/eval/{name}/decision.json", "w"), indent=1)
    if j["variant"] == "balanced":
        st["rounds"][j["round"]]["status"] = "decided_ok" if works else "decided_insufficient"
        if not works:
            st["halt_rounds"] = True
    save()
    event("round_decision", run=name, works=works,
          success=[f"{p['success_count']}/{p['eval_count']}" for p in passes], first_collapsing_stage=drop)


def run_jobs():
    if st["blocked"]:
        return
    pend = sorted([j for j in st["jobs"] if j["status"] in ("pending", "running")], key=prio)
    if not pend:
        return
    miss = env_ready()
    if miss:
        if st.get("env_wait_logged") != miss:
            st["env_wait_logged"] = miss
            save()
            event("waiting_for_sft_env", missing=miss)
        return
    j = pend[0]
    if j["kind"] == "convert":
        run_convert(j)
    elif j["kind"] == "iface":
        run_iface(j)
    elif j["kind"] == "sft":
        gate_job = next((x for x in st["jobs"] if x["round"] == j["round"] and x["kind"] == "iface"), None)
        if gate_job is None or gate_job["status"] == "done":
            run_sft(j)
    elif j["kind"] == "hsweep":
        sft_job = next(x for x in st["jobs"] if x["round"] == j["round"] and x["kind"] == "sft" and x["variant"] == j["variant"])
        if sft_job["status"] == "done":
            run_hsweep(j)
    elif j["kind"] == "eval":
        sft_job = next(x for x in st["jobs"] if x["round"] == j["round"] and x["kind"] == "sft" and x["variant"] == j["variant"])
        if sft_job["status"] == "done":
            run_eval(j)
    save()


def main():
    event("scheduler_start", rounds=ORDER)
    while True:
        try:
            ensure_collection()
            check_data()
            run_jobs()
        except Exception as e:                                     # never die silently
            event("scheduler_error", error=repr(e)[:400])
        time.sleep(60)


if __name__ == "__main__":
    main()
