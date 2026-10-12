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
* Frozen inputs: a round converts exactly the IDs of its manifest (make_manifest.py, RUNS/manifests/), never a
  scan of the growing collection. H_exec is swept on a calibration subset of the OVMM eval split that is
  disjoint (by episode) from the held-out eval subset (RUNS/eval_splits/, frozen once). RUNS/run_info/<round>.json
  records sample IDs, split, dataset version, collector/expert/reset-contract/interface commits, SFT config and
  checkpoint, H_exec decision and the calibration/eval split.
* Crash safety: one scheduler (flock); a restarted scheduler adopts the running collection supervisor,
  conversion, SFT and orphaned eval workers instead of starting duplicates (scheduler_watchdog.sh restarts it).
* CFG["stop_after_round"]: after that round's balanced eval decision no further round or job starts
  (collection continues); results stay for the user's review.

    python3 gpu_scheduler.py /workspace/lmm/configs/alphabot2_pi05/schedule.json
"""
import fcntl, hashlib, json, os, signal, subprocess, sys, time, zlib

L = os.environ.get("LMM_REPO", "/workspace/lmm")
S = f"{L}/scripts/alphabot2_habitat"
RUN = ["bash", f"{S}/run_habitat.sh"]
CFG = json.load(open(sys.argv[1]))
R = CFG["runs"]
STATE_F, EV_F = f"{R}/scheduler_state.json", f"{R}/scheduler_events.jsonl"
LOGS = CFG.get("logs", "/mnt/cfs/h0ddz1/personal/liyanjun/yanjun/lmm_runtime/logs")   # NFS, not bosfs
os.makedirs(LOGS, exist_ok=True)
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


def pgrep(pattern):
    """[(pid, cmdline)] of live processes matching the regex (never this scheduler)."""
    r = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True)
    out = []
    for l in r.stdout.splitlines():
        pid, _, cmd = l.partition(" ")
        if int(pid) != os.getpid() and "pgrep" not in cmd:
            out.append((int(pid), cmd))
    return out


def started_epoch(pid):
    e = subprocess.run(["ps", "-o", "etimes=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return time.time() - int(e) if e else None


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class Adopted:
    """A collection supervisor left running by a previous scheduler (own session; survives our restart)."""

    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        try:
            os.kill(self.pid, 0)
            return None
        except OSError:
            self.returncode = -1
            return -1


def running_supervisor():
    r = subprocess.run(["pgrep", "-f", f"{S}/collect_supervisor.py"], capture_output=True, text=True)
    pids = [int(x) for x in r.stdout.split() if int(x) != os.getpid()]
    return pids[0] if pids else None


def ensure_collection():
    p = procs.get("collect")
    if p is None and running_supervisor():                          # (None also after collection completed)
        procs["collect"] = Adopted(running_supervisor())
        event("collection_supervisor_adopted", pid=procs["collect"].pid)
        return
    if p is not None and p.poll() is None:
        return
    C = CFG["collect"]
    cs = collect_status()
    if cs.get("todo") == 0 and cs.get("processed") and os.path.getmtime(C["units_file"]) < os.path.getmtime(
            f"{C['out']}/supervisor_status.json"):
        if not st.get("collection_complete"):                      # seed bank exhausted: nothing to restart for
            st["collection_complete"] = {k: cs.get(k) for k in ("time", "processed", "accepted", "success",
                                                                "success_but_replay_mismatch")}
            save()
            event("collection_complete", **st["collection_complete"])
        procs["collect"] = None
        return
    st.pop("collection_complete", None)
    if p is not None:
        event("collection_supervisor_restart", returncode=p.returncode)
    procs["collect"] = sh(["python3", f"{S}/collect_supervisor.py"], f"{LOGS}/supervisor.log",
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
                                    "n_integrity_failures", "replay_reject_fraction")}
    st["rounds"][nxt]["last_check"] = keep
    save()
    event("data_check", round=nxt, **keep)
    if res.get("n_integrity_failures") or (res.get("replay_reject_fraction") or 0) > CFG.get("data_gate_max_replay_reject_fraction", 0.2):
        block("DATAQC", "integrity/timing/leakage failure, or replay rejections above the systemic-nondeterminism alarm",
              round=nxt, result=res)
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
def manifest_path(rnd):
    return f"{R}/manifests/{rnd}.json"


def run_convert(j):
    """Freeze the round's manifest, then convert exactly its IDs per split. A conversion still running from
    a previous scheduler is adopted (waited for); its start time is the snapshot cutoff and its train
    output becomes the manifest's train list (verified against the snapshot rule)."""
    rnd = j["round"]
    r = ROUNDS[rnd]
    live = pgrep(f"ovmm_to_lerobot.py.*--out {ds_dir(rnd, '')}")
    if live:
        if not j.get("adopted"):
            j["adopted"] = True
            tr = [p for p, c in live if "--split train" in c and "--manifest" not in c]
            if tr and not os.path.exists(manifest_path(rnd)):
                j["snapshot_cutoff"] = started_epoch(tr[0])
            event("convert_adopted", round=rnd, pids=[p for p, _ in live], snapshot_cutoff=j.get("snapshot_cutoff"))
        j["status"] = "running"
        return
    mp = manifest_path(rnd)
    if not os.path.exists(mp):
        cutoff = j.get("snapshot_cutoff") or time.time()
        j["snapshot_cutoff"] = cutoff
        idx = f"{ds_dir(rnd, 'train')}/ovmm_episode_index.json"
        cmd = ["python3", f"{S}/make_manifest.py", "--src", CFG["collect"]["out"], "--round", rnd, "--cutoff", str(cutoff),
               "--max-train", str(r["max_episodes"]), "--out", mp, "--legacy-commit", CFG["collect"].get("legacy_commit", ""),
               "--supervisor-log", f"{LOGS}/supervisor.log"]
        if j.get("adopted") and os.path.exists(idx):
            try:
                if not json.load(open(idx)).get("manifest"):
                    cmd += ["--train-from-index", idx]
            except (OSError, ValueError):
                pass
        rc = subprocess.run(cmd, stdout=open(f"{LOGS}/manifest_{rnd}.log", "a"), stderr=subprocess.STDOUT).returncode
        if rc != 0 or not os.path.exists(mp):
            j["status"] = "blocked"
            block("MANIFEST", f"make_manifest.py {rnd} exited {rc}")
            return
    m = json.load(open(mp))
    v = m.get("train_verification")
    if v and len(v["in_index_not_rule"]) + len(v["in_rule_not_index"]) > 3:
        j["status"] = "blocked"
        block("MANIFEST", "adopted train list disagrees with the snapshot rule beyond boundary noise", verification=v)
        return
    if not m["train"] or not m["val"]:
        j["status"] = "blocked"
        block("MANIFEST", "empty train or val split", n_train=m["n_train"], n_val=m["n_val"])
        return
    if not j.get("manifest_logged"):
        j["manifest_logged"] = True
        event("manifest_frozen", round=rnd, path=mp, n_train=m["n_train"], n_val=m["n_val"],
              dataset_version=m["dataset_version"], snapshot_cutoff=m["snapshot_cutoff"], train_verification=v)
    for split in ("train", "val"):
        out = ds_dir(rnd, split)
        idx = f"{out}/ovmm_episode_index.json"
        if os.path.exists(idx) and [str(e["unit"]) for e in json.load(open(idx))["episodes"]] == m[split]:
            if not os.path.exists(f"{out}/meta/exact_stats.json"):  # adopted pre-fix dataset: exact stats
                rc = subprocess.run([CFG["sft"]["python"], f"{L}/scripts/alphabot2_pi05/exact_stats.py", out],
                                    stdout=open(f"{LOGS}/exact_stats_{rnd}.log", "a"), stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    j["status"] = "blocked"
                    block("CONVERT", f"exact_stats.py {rnd} {split} exited {rc}")
                    return
                event("exact_stats_applied", round=rnd, split=split)
            continue                                               # already holds exactly the manifest's IDs
        j["status"] = "running"
        save()
        cmd = [CFG["sft"]["python"], f"{L}/scripts/alphabot2_pi05/ovmm_to_lerobot.py", "--src", CFG["collect"]["out"],
               "--manifest", mp, "--split", split, "--out", out]
        rc = subprocess.run(cmd, env={**os.environ, **env_pi()}, stdout=open(f"{LOGS}/convert_{rnd}_{split}.log", "a"),
                            stderr=subprocess.STDOUT).returncode
        ok = (rc == 0 and os.path.exists(idx) and [str(e["unit"]) for e in json.load(open(idx))["episodes"]] == m[split]
              and os.path.exists(f"{out}/meta/exact_stats.json"))
        if not ok:
            j["status"] = "blocked"
            block("CONVERT", f"ovmm_to_lerobot.py {rnd} --split {split} exited {rc} or does not match the manifest")
            return
    j["status"] = "done"
    event("converted", round=rnd, dataset_version=m["dataset_version"], n_train=m["n_train"], n_val=m["n_val"])
    write_run_info(rnd)


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
                        stdout=open(f"{LOGS}/interface_gate_{j['round']}.log", "a"), stderr=subprocess.STDOUT).returncode
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
        write_run_info(j["round"])
    else:
        j["status"] = "blocked"
        block("INTERFACE", "policy-interface gate failed", round=j["round"],
              failed=[k for k, v in (res.get("checks") or {}).items() if not v], error=res.get("error"))


def run_sft(j):
    cfg, out, name = sft_config(j["round"], j["variant"])
    p = procs.get("sft")
    if p is None:
        live = pgrep(f"sft_ovmm.py --config {cfg}")
        if live:                                                    # still running from a previous scheduler
            procs["sft"] = Adopted(min(pid for pid, _ in live))
            event("sft_adopted", run=name, pid=procs["sft"].pid)
            return
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
                           "--config", cfg], f"{LOGS}/sft_{name}_attempt{j['attempts']}.log",
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
    if os.path.exists(f"{out}/DONE") and rc in (0, -1):           # -1: adopted process (exit code unknown)
        j["status"] = "done"
        event("sft_done", run=name, summary=json.load(open(f"{out}/summary.json")))
        write_run_info(j["round"])
    elif rc == 3:
        j["status"] = "blocked"
        block("SFT", "non-finite loss (exit 3)", run=name)
    else:
        j["attempts"] = j.get("attempts", 0) + 1
        event("sft_exit_restart", run=name, returncode=rc, attempt=j["attempts"])


def val_units():
    """The AlphaBot-feasible OVMM eval-split units, or None while the background screen runs."""
    f = f"{R}/feasibility/units_{CFG['eval']['split']}.json"
    return f if os.path.exists(f) else None


def ensure_val_screen():
    """Embodiment screen of the OVMM eval split (val_screen.sh, detached, sharded by scene), started as soon
    as a round runs so it is ready when SFT finishes; co-located with collection on a GPU we may use
    (no foreign process), nothing is released for it. Gate -> units file, or a hard block."""
    E = CFG["eval"]
    f = f"{R}/feasibility/units_{E['split']}.json"
    out = f"{R}/emb_{E['split']}"
    if os.path.exists(f) or st["blocked"] or not any(x["status"] in ("pending", "running") for x in st["jobs"]):
        return
    if pgrep("val_screen.sh"):
        return
    done = f"{out}/VAL_SCREEN_DONE"
    if os.path.exists(done):
        res = gate("embodiment", out, f"{R}/feasibility/screen_{E['split']}.json", f)
        if not res.get("pass"):
            if os.path.exists(f):
                os.remove(f)
            block("EVAL_SCREEN", "eval-split embodiment screen incomplete/non-deterministic", result=res)
            return
        st["val_funnel"] = res.get("funnel_episodes")
        save()
        event("val_screen_ok", funnel=res.get("funnel_episodes"), units=res.get("units"))
        return
    g = gpus()
    ok = sorted((i for i, x in g.items() if not x["foreign"] and x["free"] >= E["server_min_free_mib"]),
                key=lambda i: -g[i]["free"])
    if not ok:
        return
    os.makedirs(out, exist_ok=True)
    sh(["bash", f"{S}/val_screen.sh", R, str(ok[0]), str(E.get("screen_workers", 4)), str(E["replicas"]), E["split"], LOGS],
       f"{LOGS}/val_screen.log")
    event("val_screen_start", gpu=ok[0], workers=E.get("screen_workers", 4), split=E["split"])


def eval_split():
    """Frozen once: calibration units (H_exec sweep) and the held-out eval units, disjoint by episode.
    Order = crc32('hexec-calib:<episode>'); the first sweep_units episodes calibrate, the rest (capped at
    eval_max_units) are the held-out normal-init eval set."""
    E = CFG["eval"]
    f = f"{R}/eval_splits/{E['split']}_calib_eval.json"
    if os.path.exists(f):
        return json.load(open(f))
    uf = val_units()
    if uf is None:
        return None
    units = json.load(open(uf))
    eps = sorted({str(u["episode_id"]) for u in units}, key=lambda e: (zlib.crc32(f"hexec-calib:{e}".encode()), e))
    cal_eps = set(eps[: E["sweep_units"]])
    rest = eps[E["sweep_units"]:]
    ev_eps = set(rest[: E["eval_max_units"]] if E.get("eval_max_units") else rest)
    cal = [u["unit"] for u in units if str(u["episode_id"]) in cal_eps]
    ev = [u["unit"] for u in units if str(u["episode_id"]) in ev_eps]
    scenes = lambda us: len({u["scene"] for u in units if u["unit"] in set(us)})
    rec = {"split": E["split"], "source": uf, "source_sha256": sha256(uf), "created": time.strftime("%F %T"),
           "rule": "episodes ordered by crc32('hexec-calib:<episode>'); first sweep_units -> calibration; next "
                   "eval_max_units -> held-out eval; disjoint by episode",
           "calibration": cal, "eval": ev, "n_calibration": len(cal), "n_eval": len(ev),
           "scenes_calibration": scenes(cal), "scenes_eval": scenes(ev), "n_feasible_units": len(units),
           "calibration_sha256": hashlib.sha256("\n".join(cal).encode()).hexdigest(),
           "eval_sha256": hashlib.sha256("\n".join(ev).encode()).hexdigest()}
    assert not ({u.split("#")[0] for u in cal} & {u.split("#")[0] for u in ev})
    os.makedirs(os.path.dirname(f), exist_ok=True)
    json.dump(rec, open(f + ".tmp", "w"), indent=1)
    os.replace(f + ".tmp", f)
    event("eval_split_frozen", path=f, n_calibration=len(cal), n_eval=len(ev),
          scenes_calibration=rec["scenes_calibration"], scenes_eval=rec["scenes_eval"])
    return rec


def h_exec():
    """The ONE execution horizon: frozen after the sanity sweep; before that the preferred value."""
    return int(st.get("h_exec_frozen") or CFG["eval"]["h_exec_preferred"])


def eval_pass(name, sft_out, units, h, seed_offset, out):
    """One end-to-end eval pass (policy server + Habitat workers on one GPU). Returns the summary, None to
    retry later (no GPU / orphaned workers of a previous scheduler still running), or {"blocked": reason}.
    Workers resume per unit; units still missing after a re-run count as engineering errors."""
    E = CFG["eval"]
    if pgrep(f"{S}/eval_policy.py"):
        if not st.get("eval_orphans_logged"):
            st["eval_orphans_logged"] = True
            event("eval_waiting_for_orphaned_workers")
        return None
    st.pop("eval_orphans_logged", None)
    srv = pgrep(f"policy_server.py --ckpt .* --port {E['port']}")
    reuse = [p for p, c in srv if f"--ckpt {sft_out}/best " in c + " "]
    for p, c in srv:                                               # our own orphaned server of another checkpoint
        if p not in reuse:
            os.killpg(p, signal.SIGTERM)
            time.sleep(10)
    g = acquire(1, 1, E["server_min_free_mib"])
    if g is None:
        return None
    gpu = g[0]
    os.makedirs(out, exist_ok=True)
    if reuse:
        server_pid = reuse[0]
        event("eval_server_adopted", pid=server_pid, out=out)
    else:
        server = sh([CFG["sft"]["python"], f"{L}/scripts/alphabot2_pi05/policy_server.py", "--ckpt", f"{sft_out}/best",
                     "--port", str(E["port"])], f"{out}/server.log", env={**env_pi(), "CUDA_VISIBLE_DEVICES": str(gpu)},
                    cwd=f"{L}/scripts/alphabot2_pi05")
        server_pid = server.pid
        t0 = time.time()
        while time.time() - t0 < 1800 and server.poll() is None and "READY" not in open(f"{out}/server.log").read():
            time.sleep(10)
        if server.poll() is not None or "READY" not in open(f"{out}/server.log").read():
            control([])
            return {"blocked": "policy server failed to start", "log": open(f"{out}/server.log").read()[-1500:]}
    for attempt in range(2):
        ws = [sh(RUN + [f"{S}/eval_policy.py"], f"{out}/worker{w}.log",
                 env={"EPISODES": ",".join(units), "SPLIT": E["split"], "PORT": str(E["port"]), "OUT": out, "GPU": str(gpu),
                      "WORKER": str(w), "NWORKERS": str(E["workers"]), "EXEC_STEPS": str(h), "SEED_OFFSET": str(seed_offset)})
              for w in range(E["workers"])]
        for w in ws:
            w.wait()
        missing = sorted(set(units) - set(r["unit"] for r in eval_rows(out)))
        if not missing:
            break
        event("eval_units_missing_rerun", out=out, missing=len(missing), attempt=attempt)
    try:
        os.killpg(server_pid, signal.SIGTERM)
    except OSError:
        pass
    control([])
    s = json.loads(subprocess.run(["python3", f"{S}/eval_summary.py", out, ",".join(units)], capture_output=True, text=True)
                   .stdout.split("###EVAL_SUMMARY###", 1)[1])
    s.update({"h_exec": h, "h_pred": CFG["eval"]["h_pred"], "seed_offset": seed_offset})
    if s["engineering_errors"] == s["eval_count"]:
        return {"blocked": "every eval episode failed with an engineering error", "summary": s}
    return s


def eval_rows(out):
    rows = []
    for i in range(100):
        f = f"{out}/eval_w{i:02d}.jsonl"
        if os.path.exists(f):
            rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def run_hsweep(j):
    """Choose the ONE execution horizon on a small fixed subset after the first balanced SFT, then freeze it."""
    E = CFG["eval"]
    _, sft_out, name = sft_config(j["round"], j["variant"])
    sp = eval_split()
    if sp is None:
        return
    units = sp["calibration"]
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
        event("hsweep_point", run=name, h_exec=h, full=f"{s['full_success_count']}/{s['eval_count']}", overall=s["overall"])
    stages = ("full", "place", "carry", "grasp", "find")
    counts = {int(h): {k: int(s["counts"][k]) for k in stages} for h, s in j["done_h"].items()}
    n = max(s["eval_count"] for s in j["done_h"].values())
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
           "calibration_split": f"{R}/eval_splits/{E['split']}_calib_eval.json", "calibration_units": units,
           "counts_by_h": counts, "h_pred": E["h_pred"], "note": "frozen for every later SFT/eval; not changed per round"}
    json.dump(rec, open(f"{R}/eval/{name}/hsweep_decision.json", "w"), indent=1)
    save()
    event("h_exec_frozen", **{k: v for k, v in rec.items() if k != "calibration_units"})
    write_run_info(j["round"])


def run_eval(j):
    E = CFG["eval"]
    _, sft_out, name = sft_config(j["round"], j["variant"])
    sp = eval_split()
    if sp is None or st.get("h_exec_frozen") is None:
        return                                                     # eval only with the frozen protocol
    units = sp["eval"]
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
        event("eval_pass_done", run=name, n=j["pass"], h_exec=s["h_exec"], success=f"{s['full_success_count']}/{s['eval_count']}",
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
        n = s["eval_count"]
        counts = {k: int(s["counts"][k]) for k in stages}
        passes.append({"seed_offset": s.get("seed_offset"), "eval_count": n, "success_count": s["full_success_count"],
                       "success_rate": round(s["full_success_count"] / max(1, n), 4),
                       "success_rate_wilson95": wilson(s["full_success_count"], n),
                       "funnel_counts": counts, "funnel_rates": {k: round(counts[k] / max(1, n), 4) for k in stages},
                       "funnel_labels": s.get("funnel_labels"), "engineering_errors": s["engineering_errors"],
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
           "eval_split": CFG["eval"]["split"], "eval_units_file": f"{R}/eval_splits/{CFG['eval']['split']}_calib_eval.json",
           "val_funnel_full_to_feasible": st.get("val_funnel"),
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
        if CFG.get("stop_after_round") == j["round"]:              # stage complete: nothing new starts
            st["halt_rounds"] = True
            st["stage_complete"] = {"round": j["round"], "time": time.strftime("%F %T"), "decision": f"{R}/eval/{name}/decision.json",
                                    "note": "stopped for the user's review; collection continues; RL/frontier never auto-start"}
            for x in st["jobs"]:
                if x["status"] in ("pending", "running") and x is not j:
                    x["status"] = "deferred"
            event("STAGE_COMPLETE", **st["stage_complete"])
    save()
    write_run_info(j["round"])
    event("round_decision", run=name, works=works,
          success=[f"{p['success_count']}/{p['eval_count']}" for p in passes], first_collapsing_stage=drop)


def write_run_info(rnd):
    try:
        _write_run_info(rnd)
    except Exception as e:                                         # reporting must never stall a job transition
        event("run_info_error", round=rnd, error=repr(e)[:300])


def _write_run_info(rnd):
    """RUNS/run_info/<round>.json (+ a copy next to the logs on CFS): everything needed to reproduce/read the round."""
    ri = {"round": rnd, "updated": time.strftime("%F %T"), "scheduler_commit": subprocess.run(
        ["git", "-C", L, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()}
    mp = manifest_path(rnd)
    if os.path.exists(mp):
        m = json.load(open(mp))
        ri["manifest"] = {"path": mp, "sha256": sha256(mp), "dataset_version": m["dataset_version"],
                          "snapshot_cutoff": m["snapshot_cutoff"], "n_train": m["n_train"], "n_val": m["n_val"],
                          "train_ids": m["train"], "val_ids": m["val"], "split_rule": m["split_rule"],
                          "train_rule": m["train_rule"], "train_verification": m.get("train_verification")}
        ri["code"] = {"collector_commits": m["collector_commits"], "expert_commit": m["code_versions"]["expert"],
                      "reset_contract_commit": m["code_versions"]["reset_contract"],
                      "policy_interface_commit": m["code_versions"]["policy_interface"], "reset_contract": m["reset_contract"]}
    ds = {}
    for split in ("train", "val"):
        idx = f"{ds_dir(rnd, split)}/ovmm_episode_index.json"
        if os.path.exists(idx):
            d = json.load(open(idx))
            ds[split] = {"root": d["root"], "episodes": len(d["episodes"]), "frames": d["frames"],
                         "keep_settle": d["keep_settle"], "manifest_sha256": d.get("manifest_sha256")}
            pio = f"{ds_dir(rnd, split)}/policy_io.json"
            if os.path.exists(pio):
                ds[split]["policy_io_sha256"] = sha256(pio)
    ri["datasets"] = ds
    gi = f"{R}/interface_gate/{rnd}/interface_gate.json"
    if os.path.exists(gi):
        g = json.load(open(gi))
        ri["policy_interface_gate"] = {"path": gi, "pass": g.get("pass"), "checks": g.get("checks")}
    ri["sft"] = {}
    for v in ROUNDS[rnd]["variants"]:
        cfg, out, name = sft_config(rnd, v)
        e = {"config": cfg, "config_sha256": sha256(cfg), "out": out, "checkpoint": f"{out}/best"}
        if os.path.exists(f"{out}/summary.json"):
            e["summary"] = json.load(open(f"{out}/summary.json"))
        ri["sft"][v] = e
    E = CFG["eval"]
    sp = f"{R}/eval_splits/{E['split']}_calib_eval.json"
    if os.path.exists(sp):
        d = json.load(open(sp))
        ri["eval_split"] = {k: d[k] for k in ("split", "source", "source_sha256", "rule", "n_calibration", "n_eval",
                                              "scenes_calibration", "scenes_eval", "calibration_sha256", "eval_sha256")}
        ri["eval_split"]["path"] = sp
    ri["h_exec"] = {"frozen": st.get("h_exec_frozen"), "h_pred": E["h_pred"], "sweep": E["h_exec_sweep"],
                    "preferred": E["h_exec_preferred"]}
    for v in ROUNDS[rnd]["variants"]:
        name = f"{rnd}_{v}"
        for k, f in (("hsweep_decision", f"{R}/eval/{name}/hsweep_decision.json"), ("eval_decision", f"{R}/eval/{name}/decision.json")):
            if os.path.exists(f):
                ri.setdefault("eval", {}).setdefault(v, {})[k] = f
    for d in (f"{R}/run_info", os.path.join(os.path.dirname(LOGS), "run_info")):
        os.makedirs(d, exist_ok=True)
        json.dump(ri, open(f"{d}/{rnd}.json.tmp", "w"), indent=1, default=str)
        os.replace(f"{d}/{rnd}.json.tmp", f"{d}/{rnd}.json")


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


def pilot_replay_check():
    """Before large-scale collection: the pilot's replay_rejected_successes / all_successes. Replay-rejected
    trajectories never enter any dataset; a high fraction means a systemic replay problem must be diagnosed
    first, so collection does not start (pilot_max_replay_reject_fraction, ~5-10%)."""
    pil = None
    try:
        for l in open(f"{R}/pipeline_status.jsonl"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if r.get("gate") == CFG.get("pilot_gate", "pilot_clean") and r.get("pass"):
                pil = r
    except OSError:
        pass
    if pil is None:
        return False, {"error": "no passed pilot gate found"}
    frac = float(pil.get("replay_reject_fraction") or 0.0)
    lim = float(CFG.get("pilot_max_replay_reject_fraction", 0.10))
    info = {"replay_rejected_successes": pil.get("replay_rejected_successes"), "replay_reject_fraction": frac,
            "limit": lim, "accepted": pil.get("complete_successes")}
    return frac <= lim, info


def main():
    lock = open("/dev/shm/lmm_gpu_scheduler.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)            # exactly one scheduler
    except OSError:
        print("another gpu_scheduler holds /dev/shm/lmm_gpu_scheduler.lock; exiting", flush=True)
        sys.exit(0)
    main.lock = lock
    event("scheduler_start", rounds=ORDER)
    ok, info = pilot_replay_check()
    event("pilot_replay_check", ok=ok, **info)
    if not ok:
        block("REPLAY_DIAGNOSIS", "pilot replay rejections above the pre-collection limit: diagnose systemic replay "
              "nondeterminism before large-scale collection (no collection started)", **info)
        while True:                                                # stay up for status; do nothing else
            time.sleep(3600)
    while True:
        try:
            ensure_collection()
            ensure_val_screen()
            check_data()
            run_jobs()
        except Exception as e:                                     # never die silently
            event("scheduler_error", error=repr(e)[:400])
        time.sleep(60)


if __name__ == "__main__":
    main()
