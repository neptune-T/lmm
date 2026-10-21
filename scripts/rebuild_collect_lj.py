"""Gated, single-process GPU7 reconstruction. Run in server tmux; never starts SFT.

Adopts the existing asset downloader by waiting for its verified receipts. Reuses the
original geometry, NAV, embodiment, manipulation, snapshot, timing and replay gates.
Every closed stage/shard is backed up before proceeding. No GPU discovery/adoption,
process killing, automatic repair, retry loop, or historical v2 manifest substitution.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
import zlib

ROOT = Path(os.environ.get("LMM_REBUILD_ROOT", "/cluster/home/gaomingju/workspace/lmm/lmm"))
CODE = ROOT / "code"
RUN = ROOT / "runtime/rebuild_20261020"
OUT = RUN / "collection"
HR = ROOT / "runtime/home-robot"
HL = HR / "src/third_party/habitat-lab"
S = CODE / "scripts/alphabot2_habitat"
PY = ROOT / "envs/home_robot39/bin/python"
P5 = ROOT / "envs/alphabot2_pi05/bin/python"
BACKUP = ROOT / "envs/backup/bin/python"
HISTORY = {"camera": "camera_top", "offsets": [5, 10, 15], "state": False}


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def event(kind, **kw):
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": kind, **kw}
    print(json.dumps(rec), flush=True)
    with (OUT / "events.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def receipt_ok(p):
    if not p.exists():
        return False
    r = json.loads(p.read_text())
    if (r.get("repo") != "Neptune-T/lmm-alphabot2-ovmm-runs" or not r.get("revision")
            or not r.get("files") or r.get("verified") != r["files"] or r.get("mismatch") != []):
        raise RuntimeError(f"Invalid verified receipt: {p}")
    return True


def resources():
    free = int(subprocess.check_output(
        ["nvidia-smi", "-i", "7", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True).strip())
    available = next(int(l.split()[1]) * 1024 for l in Path("/proc/meminfo").read_text().splitlines()
                     if l.startswith("MemAvailable:"))
    disk = shutil.disk_usage(ROOT).free
    shm = shutil.disk_usage("/dev/shm").free
    # Account for the current process's cgroup if a finite limit is present.
    cg_remaining = None
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            cg = Path("/sys/fs/cgroup") / line.split("::", 1)[1].lstrip("/")
            if (cg / "memory.max").exists():
                limit = (cg / "memory.max").read_text().strip()
                if limit != "max":
                    cg_remaining = int(limit) - int((cg / "memory.current").read_text())
                    available = min(available, cg_remaining)
    return {"gpu": 7, "free_mib": free, "ram_available": available,
            "persistent_free": disk, "shm_free": shm, "cgroup_remaining": cg_remaining}


def capacity():
    r = resources()
    event("RESOURCE_CHECK", **r)
    if r["free_mib"] < 8192 or r["ram_available"] < 32 * 2**30 or r["persistent_free"] < 100 * 2**30:
        raise RuntimeError("Insufficient GPU7/RAM/disk headroom for one simulator; other jobs left untouched")


def environment(extra=None):
    e = {**os.environ, "LMM_REPO": str(CODE), "GPU": "7", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
         "ALPHABOT2_OVMM_CONFIG": str(HL / "habitat-lab/habitat/config/benchmark/ovmm/ovmm_alphabot2.yaml"),
         "ALPHABOT2_ASSETS": str(ROOT / "runtime/alphabot2_assets"),
         "PYTHONPATH": f"{HL}/habitat-lab:{HL}/habitat-baselines:{HR}/src:{CODE}",
         "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
         "PYTHONUNBUFFERED": "1", "HF_HOME": str(ROOT / "hf"),
         "HF_TOKEN_PATH": str(Path.home() / ".config/lmm/hf_token"),
         "HF_SYNC_STAGE": str(ROOT / "backups/stage"),
         "PI05_CKPT": str(ROOT / "runtime/pi05/ckpt"),
         "PI05_SPM": str(ROOT / "runtime/pi05/paligemma_tokenizer.model"),
         "ALPHABOT2_LOG_DIR": str(ROOT / "logs"),
         "TORCHDYNAMO_DISABLE": "1"}
    # Habitat EGL takes a PHYSICAL device index; this simulator env has CPU-only torch.
    e.pop("CUDA_VISIBLE_DEVICES", None)
    for key in ("EPISODES", "EPISODE_FILE", "CLAIM_DIR", "MAX_OK", "NAV_HEIGHT",
                "NAV_RADIUS", "ALPHABOT2_NAVMESH_TAG", "SMOKE_EPISODE"):
        e.pop(key, None)
    e.update({k: str(v) for k, v in (extra or {}).items()})
    return e


def run(stage, label, command, extra=None, gpu=False):
    if gpu:
        capacity()
    log = stage / f"{label}.log"
    event("COMMAND_START", stage=stage.name, command=label, gpu=7 if gpu else None)
    with log.open("w") as f:
        rc = subprocess.call(["/usr/bin/time", "-v", "-o", str(stage / f"{label}.resources.txt"),
                              *map(str, command)], cwd=HR, env=environment(extra),
                             stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
    if rc:
        raise RuntimeError(f"{label} exited {rc}; see {log}")


def sim(stage, label, script, extra=None):
    run(stage, label, [PY, S / script], extra, gpu=True)


def gate(stage, name, *args):
    run(stage, f"gate_{name}", [PY, S / "gates.py", name, *args])


def backup(stage):
    receipt = OUT / "receipts" / f"{stage.name}.json"
    with (OUT / "backup.log").open("a") as log:
        subprocess.run([str(BACKUP), str(CODE / "scripts/hf_sync.py"),
                        "--tag", f"lj_recollect_20261020_{stage.name}", str(stage),
                        "--receipt", str(receipt)], env=environment(),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    assert receipt_ok(receipt)


def stage(name, fn):
    d = OUT / "stages" / name
    result = d / "result.json"
    if result.exists() and json.loads(result.read_text()).get("pass"):
        if not receipt_ok(OUT / "receipts" / f"{name}.json"):
            backup(d)  # completed work is not rerun when only backup was interrupted
        return d
    # Preserve interrupted/failed outputs. No silent retries or deletion of old runs.
    if d.exists():
        raise RuntimeError(f"Unfinished stage needs diagnosis before resuming: {d}")
    d.mkdir(parents=True)
    event("STAGE_START", stage=name)
    try:
        fn(d)
    except Exception:
        dump(result, {"pass": False, "traceback": traceback.format_exc()})
        backup(d)
        raise
    dump(result, {"pass": True, "code_commit": COMMIT, "finished": time.time()})
    backup(d)
    event("STAGE_BACKUP_VERIFIED", stage=name)
    return d


def rows(d, pattern="results*.jsonl"):
    return [json.loads(line) for p in sorted(Path(d).glob(pattern)) for line in p.read_text().splitlines() if line]


def split_of(e):
    return "val" if zlib.crc32(f"sft-split:{e}".encode()) % 1000 < 50 else "train"


def scene_order(items):
    """Fixed round-robin geometry pool, before observing any expert outcomes."""
    by = {}
    rng = random.Random(1000)
    for r in sorted(items, key=lambda r: str(r["episode_id"])):
        by.setdefault(r["scene"], []).append(r)
    for pool in by.values():
        rng.shuffle(pool)
    scenes = sorted(by)
    rng.shuffle(scenes)
    out = []
    while any(by.values()):
        for sc in scenes:
            if by[sc]:
                out.append(by[sc].pop())
    return out


def qualification_lists(units):
    """Reserve disjoint train bases; readiness never depends on rollout outcomes."""
    candidates = [u for u in units if u["replica"] == 0 and split_of(u["episode_id"]) == "train"]
    used, result = set(), {}
    for mode, n in (("left", 5), ("right", 5), ("auto", 8)):
        pick = [u for u in candidates if u["episode_id"] not in used
                and (mode == "auto" or mode in u["feasible_arms"])][:n]
        if len(pick) != n:
            return None
        used.update(u["episode_id"] for u in pick)
        result[f"qual_{mode}"] = pick
    pilot = [u for u in candidates if u["episode_id"] not in used][:80]
    if len(pilot) < 40:
        return None
    return {**result, "pilot": pilot, "excluded_qualification_bases": sorted(used)}


def reject_engineering(d):
    bad = [r for r in rows(d) if r.get("engineering_error")]
    if bad:
        raise RuntimeError(f"Engineering errors in {len(bad)} episodes; first={bad[0]['episode_id']}")


def main():
    global COMMIT
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "receipts").mkdir(exist_ok=True)
    lock = (OUT / "driver.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    COMMIT = subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip()
    event("START", code_commit=COMMIT, scope="reconstruction checks and collection only; no training")

    def cpu(d):
        run(d, "contracts", [P5, "-m", "unittest",
             "tests.test_alphabot2_ovmm_policy_io", "tests.test_alphabot2_ovmm_chain_check",
             "tests.test_alphabot2_ovmm_sft_windows", "tests.test_alphabot2_habitat_urdf"],
            {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(CODE)})
        run(d, "pi05_history", [P5, CODE / "scripts/rebuild_history_check.py"],
            {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(CODE)})
        dump(d / "protocol.json", {
            "run_id": "lj_recollect_20261020", "original_v2_recovered": False,
            "gpu": 7, "sim_workers": 1, "seed": 1000, "history": HISTORY,
            "state_dim": 16, "action_dim": 18, "history_seconds": [0.5, 1.0, 1.5],
            "screen_replicas": 3, "initial_screen_base_episodes": 256, "screen_shard_size": 64,
            "collection_shard_units": 8, "pilot_max_accepted": 15,
            "pilot_gate": "existing gates.py pilot: 10-20 accepted; both arms; >1 scene",
            "split_rule": "crc32('sft-split:<base episode>') % 1000 < 50 -> SFT val",
            "final_eval": "OVMM-val excluded from collection",
            "stop_target": {"train_accepted": 2000, "val_accepted": 100},
            "training_launched": False,
            "deferred_training_gates": ["all-video decode/drop/alignment", "exact train-only stats",
                "real-data model/history/interface gate", "measured training VRAM and resume gate"],
            "code_commit": COMMIT})
    stage("cpu_preflight", cpu)

    required = ["base_model", "pi05_env", "habitat_env", "robot", "robot_load", "episodes", "objects", "scenes"]
    event("WAIT_ASSETS", required=required)
    wait_start = time.time()
    while not all(receipt_ok(RUN / f"{n}_backup_verified.json") for n in required):
        # This is a server-side dependency wait, not a replacement downloader.
        if time.time() - wait_start > 24 * 3600:
            raise RuntimeError("Assets still unverified after 24 hours; inspect downloader without restarting it")
        time.sleep(60)
    def assets(d):
        for name in required:
            shutil.copy2(RUN / f"{name}_backup_verified.json", d)
        for name, rel in (("episodes", "datasets/ovmm"), ("objects", "objects"), ("scenes", "hssd-hab")):
            m = json.loads((RUN / f"{name}_upstream_manifest.json").read_text())
            for rec in m["files"]:
                p = HR / "data" / rel / rec["path"]
                if not p.is_file() or p.stat().st_size != rec["size"]:
                    raise RuntimeError(f"Asset missing/changed since upstream verification: {p}")
                h = hashlib.sha256()
                with p.open("rb") as f:
                    for b in iter(lambda: f.read(4 << 20), b""):
                        h.update(b)
                if h.hexdigest() != rec["sha256"]:
                    raise RuntimeError(f"Asset hash changed since verified backup: {p}")
            shutil.copy2(RUN / f"{name}_upstream_manifest.json", d)
        dump(d / "resources.json", resources())
    stage("assets_ready", assets)
    def geometry(d):
        sim(d, "screen", "feasibility_screen.py", {"OUT": d, "SPLITS": "train,val"})
        gate(d, "screen", d / "summary.json", CODE / "artifacts/alphabot2_habitat/feasibility_summary.json")
        pool = scene_order([r for r in json.loads((d / "screen_train.json").read_text()) if r["feasible"]])
        dump(d / "frozen_pool.json", [{"episode_id": r["episode_id"], "scene": r["scene"],
                                      "sft_split": split_of(r["episode_id"])} for r in pool])
        dump(d / "ordered_geometry.json", pool)
    geo = stage("geometry", geometry)
    pool = json.loads((geo / "ordered_geometry.json").read_text())

    def nav(d):
        for tag, extra in (("h17", {"NAV_HEIGHT": "1.7", "ALPHABOT2_NAVMESH_TAG": "alphabot2_r038_h17"}),
                           ("h15", {})):
            sim(d, tag, "nav_qual.py", {"OUT": d / tag, "SPLIT": "val", **extra})
        gate(d, "navmesh", d / "h17", d / "h15")
    navdir = stage("nav_audit", nav)
    reachable = [r["episode_id"] for r in rows(navdir / "h15") if not r.get("engineering_error")
                 and r.get("to_object", {}).get("arrived") and r.get("to_receptacle", {}).get("arrived")]
    if not reachable:
        raise RuntimeError("No reachable qualification episode")
    def smoke(d):
        sim(d, "smoke", "smoke.py", {"SPLIT": "val", "SMOKE_EPISODE": reachable[0], "SMOKE_OUT": d})
        gate(d, "smoke", d / "smoke.json")
        sim(d, "timing", "timing_sync.py", {"SPLIT": "val", "EPISODE": reachable[0], "CONCUR_RENDER": "0"})
        reports = [json.loads(l.split("###SYNC###", 1)[1]) for l in (d / "timing.log").read_text().splitlines()
                   if l.startswith("###SYNC###")]
        if len(reports) != 1 or reports[0]["synchronized"] is not True:
            raise RuntimeError("RGB/proprio/action timing gate failed")
        dump(d / "timing.json", reports[0])
    stage("smoke_timing", smoke)

    def screen_chunk(i, rs):
        def work(d):
            dump(d / "input_geometry.json", rs)
            sim(d, "emb", "emb_screen.py", {"SCREEN": d / "input_geometry.json", "OUT": d / "emb",
                "REPLICAS": 3, "DETERMINISM_EVERY": 1, "SPLIT": "train"})
            gate(d, "embodiment", d / "emb", d / "input_geometry.json", d / "units.json")
        return stage(f"screen_{i:04d}", work)
    units = []
    # Four shards were an unverified capacity estimate, not a data-quality gate.
    # Continue the same frozen order until existing disjoint-base minima are met.
    for i in range((len(pool) + 63) // 64):
        sd = screen_chunk(i, pool[i * 64:(i + 1) * 64])
        units += json.loads((sd / "units.json").read_text())
        if i >= 3 and qualification_lists(units) is not None:
            break

    def lists(d):
        selected = qualification_lists(units)
        if selected is None:
            raise RuntimeError("Frozen geometry pool exhausted before qualification/pilot minima")
        for name, value in selected.items():
            dump(d / f"{name}.json", value)
    ld = stage("lists_frozen", lists)
    excluded = set(json.loads((ld / "excluded_qualification_bases.json").read_text()))

    def manipulation(d):
        for mode in ("left", "right", "auto"):
            eps = ",".join(u["episode_id"] for u in json.loads((ld / f"qual_{mode}.json").read_text()))
            sim(d, mode, "manip_qual.py", {"OUT": d, "SPLIT": "train", "MODES": mode,
                                         "RESULTS_TAG": mode, "EPISODES": eps})
        reject_engineering(d)
        gate(d, "manip", d)
    md = stage("manipulation", manipulation)
    def snapshot(d):
        eps = [r["episode_id"] for r in rows(md) if r["mode"] == "auto" and r.get("expert_completed")
               and r.get("ovmm", {}).get("ovmm_place_success") == 1][:4]
        sim(d, "snapshot", "snapshot_qual.py", {"OUT": d, "SPLIT": "train", "EPISODES": ",".join(eps)})
        gate(d, "snapshot", d)
    stage("snapshot", snapshot)

    collected = []
    def collect(name, us, max_ok=None):
        def work(d):
            dump(d / "units.json", us)
            extra = {"EPISODES": ",".join(u["unit"] for u in us), "OUT": d / "data",
                     "WORKER": 0, "NWORKERS": 1}
            if max_ok is not None:
                extra["MAX_OK"] = max_ok
            sim(d, "collector", "pilot.py", extra)
            reject_engineering(d / "data")
            # Actual labels, camera shapes, finite values, timestamps and historical source indices.
            run(d, "raw_check", [P5, CODE / "scripts/rebuild_history_check.py", d / "data"],
                {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(CODE)})
        p = stage(name, work)
        collected.append(p / "data")
        return rows(p / "data")

    pilot_units = json.loads((ld / "pilot.json").read_text())
    pilot_rows = []
    for i in range(0, len(pilot_units), 4):
        if sum(bool(r.get("accepted")) for r in pilot_rows) >= 15:
            break
        pilot_rows += collect(f"pilot_{i // 4:04d}", pilot_units[i:i + 4],
                              15 - sum(bool(r.get("accepted")) for r in pilot_rows))
    def pilot_gate(d):
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pilot_rows))
        (d / "units.txt").write_text("\n".join(u["unit"] for u in pilot_units))
        gate(d, "pilot", d, d / "units.txt")
    stage("pilot_gate", pilot_gate)
    done = {r["episode_id"] for r in pilot_rows}
    all_rows = list(pilot_rows)
    def counts():
        return {sp: sum(bool(r.get("accepted")) and split_of(r["episode"]) == sp for r in all_rows)
                for sp in ("train", "val")}
    def target():
        c = counts()
        return c["train"] >= 2000 and c["val"] >= 100
    # Complete the initial screened pool, then screen/collect one closed shard at a time.
    for i in range((len(pool) + 63) // 64):
        if target():
            break
        sd = screen_chunk(i, pool[i * 64:(i + 1) * 64])
        us = [u for u in json.loads((sd / "units.json").read_text())
              if u["episode_id"] not in excluded and u["unit"] not in done]
        for j in range(0, len(us), 8):
            if target():
                break
            rs = collect(f"collect_{i:04d}_{j // 8:03d}", us[j:j + 8])
            all_rows += rs
            done.update(r["episode_id"] for r in rs)
            event("COLLECTION_PROGRESS", accepted=counts(), attempted=len(all_rows))
    def finish(d):
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_rows))
        dump(d / "collection_index.json", {"run_id": "lj_recollect_20261020",
            "shards": list(map(str, collected)), "accepted": counts(), "target_reached": target(),
            "excluded_qualification_bases": sorted(excluded), "training_started": False})
        (d / "excluded.txt").write_text("\n".join(sorted(excluded)))
        run(d, "manifest", [BACKUP, S / "make_manifest.py", "--src", d,
            "--round", "lj_recollect_20261020", "--cutoff", time.time(),
            "--exclude-episodes", d / "excluded.txt", "--out", d / "manifest.json"])
    stage("collection_index", finish)
    event("COLLECTION_COMPLETE" if target() else "POOL_EXHAUSTED", accepted=counts(),
          next="full data QC, new frozen manifest, conversion and actual training gates; no automatic training")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if OUT.exists():
            event("BLOCKED", error=str(e))
        raise
