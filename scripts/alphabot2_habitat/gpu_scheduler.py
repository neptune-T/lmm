"""Long-running AlphaBot 2 OVMM scheduler (system python3, stdlib only):
continuous collection -> data QC gate -> LeRobot conversion -> pi0.5 SFT -> normal-init e2e eval.

* Collection always runs (collect_supervisor.py child, restarted if it dies; it adopts idle GPUs).
* GPUs: a GPU is usable only if no foreign compute process is on it (a compute-app pid not visible in
  this container). SFT takes truly idle GPUs first, then collection GPUs beyond min_collect_gpus, which
  are handed back through collect_control.json. Other users' processes are never touched or preempted.
* Hard gates stop every later stage and are recorded with their reason; nothing is relaxed:
  data QC (gates.py dataqc), conversion failure, SFT non-finite loss (exit 3), SFT restarts exhausted,
  val feasibility screen incomplete, eval engineering failure of every episode.
  Missing SFT env/assets is a recorded block that is re-checked each poll (collection keeps going).
* SFT crash/stall -> restart (resume from the newest checkpoint) up to max_restarts.
* After eval: funnel overall / by hand / by scene; decision.json states which branch applies
  (full-task success stable non-zero -> next: empirical frontier + FP8 rollout scaling; ~0 -> analyse
  funnel and SFT/data coverage). The scheduler never starts RL.
State: RUNS/scheduler_state.json; events: RUNS/scheduler_events.jsonl.

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
st = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {"stage": "COLLECTING", "sft_attempts": 0}
procs = {}


def save():
    json.dump(st, open(STATE_F + ".tmp", "w"), indent=1)
    os.replace(STATE_F + ".tmp", STATE_F)


def event(kind, **kw):
    rec = {"time": time.strftime("%F %T"), "event": kind, "stage": st["stage"], **kw}
    with open(EV_F, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def block(stage, reason, **kw):
    st["stage"], st["blocked"] = f"BLOCKED_{stage}", {"reason": reason, **kw}
    save()
    event("BLOCKED", gate=stage, reason=reason, **kw)


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
    out = []
    for l in q.strip().splitlines():
        i, uuid, used, tot, util = [x.strip() for x in l.split(",")]
        out.append({"index": int(i), "used": int(used), "free": int(tot) - int(used), "util": int(util),
                    "foreign": uuid in foreign})
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


def pick_gpus(n, min_free):
    """n GPUs with no foreign process and >= min_free MiB free once our collection leaves them."""
    owned = set(collect_status().get("owned_gpus", []))
    keep = CFG["collect"]["min_collect_gpus"]
    gs = [g for g in gpus() if not g["foreign"]]
    idle = [g["index"] for g in gs if g["index"] not in owned and g["used"] < 500 and g["free"] >= min_free]
    ours = [g["index"] for g in gs if g["index"] in owned]
    spare = ours[keep:] if len(ours) > keep else []
    chosen = (idle + spare)[:n]
    return chosen if len(chosen) == n else None


def sft_cfg():
    out, name = {}, None
    for l in open(CFG["sft"]["config"]):                  # tiny yaml subset: top-level scalar keys
        if ":" in l and not l.startswith((" ", "#")):
            k, v = l.split(":", 1)
            out[k.strip()] = v.split("#")[0].strip()
    return out


def step():
    stg = st["stage"]
    if stg.startswith("BLOCKED") and not stg == "BLOCKED_SFT_ENV":
        return
    if stg in ("COLLECTING",):
        if time.time() - st.get("last_qc", 0) < CFG["data_gate"]["check_every_s"]:
            return
        st["last_qc"] = time.time()
        D = CFG["data_gate"]
        res = gate("dataqc", CFG["collect"]["out"], f"{R}/feasibility/units_train.json", str(D["min_accepted"]),
                   str(D["min_arm_share"]), str(D["min_scene_coverage"]))
        st["last_qc_result"] = {k: res.get(k) for k in ("pass", "accepted", "arm_share", "scene_coverage",
                                                        "n_integrity_failures", "success_but_replay_mismatch")}
        save()
        event("dataqc", **st["last_qc_result"])
        if res.get("n_integrity_failures") or res.get("success_but_replay_mismatch"):
            block("DATAQC", "integrity/timing/leakage failure or determinism break in accepted data", result=res)
        elif res.get("pass"):
            st["stage"] = "SFT_ENV"
            save()
        return
    if stg in ("SFT_ENV", "BLOCKED_SFT_ENV"):
        A, py = CFG["sft"]["assets"], CFG["sft"]["python"]
        missing = [p for p in (py, f"{A}/ckpt/config.json", f"{A}/paligemma_tokenizer.model") if not os.path.exists(p)]
        if missing:
            if stg != "BLOCKED_SFT_ENV":
                block("SFT_ENV", "pi0.5 env or assets missing (collection continues; re-checked every poll)", missing=missing)
            return
        st["stage"] = "CONVERT"
        save()
        event("sft_env_ok")
        return
    sc = sft_cfg()
    env_pi = {"PI05_CKPT": f"{CFG['sft']['assets']}/ckpt", "PI05_SPM": f"{CFG['sft']['assets']}/paligemma_tokenizer.model",
              "TORCHDYNAMO_DISABLE": "1", "LMM_REPO": L, "PYTHONUNBUFFERED": "1"}
    if stg == "CONVERT":
        for split, key in (("train", "train_data"), ("val", "val_data")):
            r = subprocess.run([CFG["sft"]["python"], f"{L}/scripts/alphabot2_pi05/ovmm_to_lerobot.py", "--src", CFG["collect"]["out"],
                                "--split", split, "--out", sc[key]], env={**os.environ, **env_pi},
                               stdout=open(f"{R}/logs/convert_{split}.log", "a"), stderr=subprocess.STDOUT)
            if r.returncode != 0:
                block("CONVERT", f"ovmm_to_lerobot.py --split {split} exited {r.returncode}")
                return
        st["stage"] = "SFT"
        save()
        event("converted")
        return
    if stg == "SFT":
        p = procs.get("sft")
        out = sc["out"]
        if p is None or p.poll() is not None:
            if p is not None:
                rc = p.returncode
                if rc == 0 and os.path.exists(f"{out}/DONE"):
                    control([])
                    st["stage"] = "EVAL_SCREEN"
                    save()
                    event("sft_done", summary=json.load(open(f"{out}/summary.json")))
                    return
                if rc == 3:
                    block("SFT", "non-finite loss (exit 3)")
                    return
                st["sft_attempts"] += 1
                event("sft_exit", returncode=rc, attempt=st["sft_attempts"])
                if st["sft_attempts"] > CFG["sft"]["max_restarts"]:
                    block("SFT", "restarts exhausted", returncode=rc)
                    return
            chosen = pick_gpus(CFG["sft"]["gpus"], CFG["sft"]["min_free_mib"])
            if chosen is None:
                return                                             # wait for free GPUs; collection keeps running
            control(chosen)
            time.sleep(CFG["collect"]["poll_s"] + 30)              # let the supervisor hand them back
            now = {g["index"]: g for g in gpus()}
            if any(now[i]["foreign"] or now[i]["free"] < CFG["sft"]["min_free_mib"] for i in chosen):
                control([])                                        # someone else got there first: retry later
                event("sft_gpus_not_free", gpus=chosen)
                return
            procs["sft"] = sh([CFG["sft"]["python"], "-m", "torch.distributed.run", f"--nproc_per_node={len(chosen)}",
                               f"--master_port={CFG['sft']['master_port']}", f"{L}/scripts/alphabot2_pi05/sft_ovmm.py",
                               "--config", CFG["sft"]["config"]], f"{R}/logs/sft_attempt{st['sft_attempts']}.log",
                              env={**env_pi, "CUDA_VISIBLE_DEVICES": ",".join(map(str, chosen))}, cwd=f"{L}/scripts/alphabot2_pi05")
            st["sft_gpus"] = chosen
            save()
            event("sft_launch", gpus=chosen, attempt=st["sft_attempts"])
            return
        try:
            hb = json.load(open(f"{out}/heartbeat.json"))
            stalled = time.time() - hb["time"] > CFG["sft"]["stall_s"]
        except (OSError, ValueError):
            stalled = False
        if stalled:
            event("sft_stall_kill", heartbeat=hb)
            os.killpg(p.pid, signal.SIGTERM)
        return
    if stg == "EVAL_SCREEN":
        E = CFG["eval"]
        g = pick_gpus(1, E["server_min_free_mib"])
        if g is None:
            return
        subprocess.run(RUN + [f"{S}/emb_screen.py"], env={**os.environ, "SCREEN": f"{R}/feasibility/screen_{E['split']}.json",
                                                         "SPLIT": E["split"], "REPLICAS": str(E["replicas"]), "GPU": str(g[0]),
                                                         "OUT": f"{R}/emb_{E['split']}"},
                       stdout=open(f"{R}/logs/emb_{E['split']}.log", "a"), stderr=subprocess.STDOUT)
        res = gate("embodiment", f"{R}/emb_{E['split']}", f"{R}/feasibility/screen_{E['split']}.json",
                   f"{R}/feasibility/units_{E['split']}.json")
        if not res.get("pass"):
            block("EVAL_SCREEN", "val embodiment screen incomplete/non-deterministic", result=res)
            return
        st["eval_units"] = len(json.load(open(f"{R}/feasibility/units_{E['split']}.json")))
        st["eval_funnel_val"] = res.get("funnel_episodes")
        st["stage"], st["eval_pass"] = "EVAL", 0
        save()
        event("eval_screen_ok", units=st["eval_units"], funnel=res.get("funnel_episodes"))
        return
    if stg == "EVAL":
        E = CFG["eval"]
        if st["eval_pass"] >= E["passes"]:
            st["stage"] = "REPORT"
            save()
            return
        g = pick_gpus(1, E["server_min_free_mib"])
        if g is None:
            return
        gpu = g[0]
        control([gpu])
        out = f"{R}/eval/{sc['name']}/pass{st['eval_pass']}"
        os.makedirs(out, exist_ok=True)
        server = sh([CFG["sft"]["python"], f"{L}/scripts/alphabot2_pi05/policy_server.py", "--ckpt", f"{sc['out']}/best",
                     "--port", str(E["port"])], f"{out}/server.log", env={**env_pi, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    cwd=f"{L}/scripts/alphabot2_pi05")
        t0 = time.time()
        while time.time() - t0 < 1800 and "READY" not in open(f"{out}/server.log").read():
            if server.poll() is not None:
                break
            time.sleep(10)
        if server.poll() is not None or "READY" not in open(f"{out}/server.log").read():
            control([])
            block("EVAL", "policy server failed to start", log=open(f"{out}/server.log").read()[-1500:])
            return
        units = ",".join(u["unit"] for u in json.load(open(f"{R}/feasibility/units_{E['split']}.json")))
        ws = [sh(RUN + [f"{S}/eval_policy.py"], f"{out}/worker{w}.log",
                 env={"EPISODES": units, "SPLIT": E["split"], "PORT": str(E["port"]), "OUT": out, "GPU": str(gpu),
                      "WORKER": str(w), "NWORKERS": str(E["workers"]), "EXEC_STEPS": str(E["exec_steps"]),
                      "SEED_OFFSET": str(st["eval_pass"])}) for w in range(E["workers"])]
        for w in ws:
            w.wait()
        os.killpg(server.pid, signal.SIGTERM)
        control([])
        summ = json.loads(subprocess.run(["python3", f"{S}/eval_summary.py", out], capture_output=True, text=True)
                          .stdout.split("###EVAL_SUMMARY###", 1)[1])
        if summ["engineering_errors"] == summ["episodes"]:
            block("EVAL", "every eval episode failed with an engineering error", summary=summ)
            return
        st.setdefault("eval_summaries", []).append(summ)
        st["eval_pass"] += 1
        save()
        event("eval_pass_done", n=st["eval_pass"], overall=summ["overall"], full_success_count=summ["full_success_count"])
        return
    if stg == "REPORT":
        ss = st["eval_summaries"]
        full = [s["full_success_count"] for s in ss]
        stable = all(f > 0 for f in full)
        dec = {"sft": sc["name"], "eval_split": CFG["eval"]["split"], "eval_units": st.get("eval_units"),
               "val_funnel_full_to_feasible": st.get("eval_funnel_val"),
               "passes": [{"overall": s["overall"], "full_success_count": s["full_success_count"],
                           "by_hand": s["by_hand"]} for s in ss],
               "full_success_stable_nonzero": stable,
               "next": ("empirical frontier + FP8 rollout scaling" if stable else
                        "analyse the funnel and SFT/data coverage; do not enter RL"),
               "note": "evaluated on the AlphaBot-feasible val subset; not standard OVMM; the scheduler starts no RL"}
        json.dump(dec, open(f"{R}/eval/{sc['name']}/decision.json", "w"), indent=1)
        st["stage"] = "DONE_ROUND1"
        save()
        event("round1_decision", **{k: dec[k] for k in ("full_success_stable_nonzero", "next")})


def main():
    event("scheduler_start")
    while True:
        try:
            ensure_collection()
            step()
        except Exception as e:                                     # never let the scheduler die silently
            event("scheduler_error", error=repr(e)[:300])
        time.sleep(60)


if __name__ == "__main__":
    main()
