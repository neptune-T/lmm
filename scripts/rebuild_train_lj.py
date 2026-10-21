"""User-authorized B reconstruction: adopt collection, gate data, then GPU7 SFT.

Never starts/restarts the collector, deletes failed stages, signals foreign processes,
or uses old v2 manifests. One CPU conversion shard and one GPU job at a time.
Every closed stage/checkpoint must have a verified private HF backup before proceeding.
"""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import rebuild_collect_lj as collection

ROOT, CODE, RUN = collection.ROOT, collection.CODE, collection.RUN
OUT = RUN / "training"
P5 = ROOT / "envs/alphabot2_pi05/bin/python"
BACKUP = ROOT / "envs/backup/bin/python"
S = CODE / "scripts/alphabot2_pi05"
COL = RUN / "collection"
MODEL = OUT / "model_B"
COMMIT = None
dump = collection.dump
receipt_ok = collection.receipt_ok


def require_training_authorized():
    if (RUN / "TRAINING_PAUSED_BY_USER.json").exists():
        raise RuntimeError("Training paused by user: data recovery only; new authorization required")


def event(kind, **kw):
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": kind, **kw}
    print(json.dumps(rec), flush=True)
    with (OUT / "events.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def env(gpu=False):
    e = collection.environment()
    e.update(CUDA_VISIBLE_DEVICES="7" if gpu else "", PYTHONPATH=str(CODE),
             PATH=f"{P5.parent}:{os.environ.get('PATH', '')}")
    # Reconstructed run does not inherit distributed launcher settings.
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        e.pop(k, None)
    return e


def require_capacity(gpu=False):
    r = collection.resources()
    # Conservative initial budget; the two actual optimizer steps measure feasibility.
    if r["ram_available"] < 96 * 2**30 or r["persistent_free"] < 500 * 2**30:
        raise RuntimeError(f"Insufficient CPU RAM/persistent disk: {r}")
    if gpu and r["free_mib"] < 100 * 1024:
        return False, r
    return True, r


def wait_gpu():
    announced = False
    while True:
        require_training_authorized()
        ready, r = require_capacity(True)
        if ready:
            event("GPU7_HEADROOM", **r)
            return
        if not announced:
            event("WAIT_GPU7_HEADROOM", **r)
            announced = True
        time.sleep(60)


def command(d, name, argv, gpu=False):
    require_training_authorized()
    if gpu:
        wait_gpu()
    else:
        require_capacity()
    event("COMMAND_START", stage=d.name, command=name, gpu=7 if gpu else None)
    with (d / f"{name}.log").open("w") as log, (d / f"{name}.memory.jsonl").open("w") as mem:
        proc = subprocess.Popen(["/usr/bin/time", "-v", "-o", str(d / f"{name}.resources.txt"),
                                 *map(str, argv)], cwd=CODE, env=env(gpu),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            # Observational only: never kills a process on a resource warning.
            mem.write(json.dumps({"time": time.time(), **collection.resources()}) + "\n")
            mem.flush()
            time.sleep(10)
        if proc.returncode:
            raise RuntimeError(f"{name} exited {proc.returncode}: {d / (name + '.log')}")


def backup(d, extra=()):
    receipt = OUT / "receipts" / f"{d.name}.json"
    with (OUT / "backup.log").open("a") as log:
        subprocess.run([str(BACKUP), str(CODE / "scripts/hf_sync.py"),
                        "--tag", f"lj_retrain_20261020_{d.name}", str(d), *map(str, extra),
                        "--receipt", str(receipt)], env=env(),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if not receipt_ok(receipt):
        raise RuntimeError(f"Backup not verified: {receipt}")


def stage(name, fn, extra=()):
    d = OUT / "stages" / name
    result = d / "result.json"
    if result.exists() and json.loads(result.read_text()).get("pass") is True:
        if not receipt_ok(OUT / "receipts" / f"{name}.json"):
            backup(d, extra)
        return d
    if d.exists():
        raise RuntimeError(f"Preserved incomplete stage requires diagnosis: {d}")
    d.mkdir(parents=True)
    event("STAGE_START", stage=name)
    try:
        fn(d)
    except Exception:
        dump(result, {"pass": False, "traceback": traceback.format_exc()})
        backup(d)
        raise
    dump(result, {"pass": True, "code_commit": COMMIT, "finished": time.time()})
    backup(d, extra)
    event("STAGE_BACKUP_VERIFIED", stage=name)
    return d


def validate_manifest(m, index):
    if not index.get("target_reached") or m["n_train"] < 2000 or m["n_val"] < 100:
        raise RuntimeError("Collection target not met; do not train an undersized substitute")
    if not m["dataset_version"].startswith("lj_recollect_20261020-"):
        raise RuntimeError("Expected a new reconstruction manifest")
    for split in ("train", "val"):
        if len(m[split]) != m[f"n_{split}"] or len(set(m[split])) != len(m[split]):
            raise RuntimeError(f"Manifest count/duplicate error: {split}")
        details = m[f"{split}_detail"]
        if [r["unit"] for r in details] != m[split] or not all(r["code_commit"] for r in details):
            raise RuntimeError(f"Manifest provenance error: {split}")
    bases = [{r["episode"] for r in m[f"{sp}_detail"]} for sp in ("train", "val")]
    if bases[0] & bases[1] or (bases[0] | bases[1]) & set(m["excluded_episodes"]):
        raise RuntimeError("Base-episode leakage or qualification contamination")


def passed(path):
    r = json.loads(Path(path).read_text())
    if r.get("pass") is not True or not all(r.get("checks", {}).values()):
        raise RuntimeError(f"Gate failed: {path}")


def collector_alive():
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid == os.getuid():
                args = (p / "cmdline").read_bytes().decode().split("\0")
                if any(a in ("code/scripts/rebuild_collect_lj.py",
                             str(CODE / "scripts/rebuild_collect_lj.py")) for a in args):
                    return True
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return False


def main():
    global COMMIT
    require_training_authorized()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "driver.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUT / "receipts").mkdir(exist_ok=True)
    COMMIT = subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip()
    event("START", code_commit=COMMIT, scope="new reconstructed B baseline, 10000 steps; GPU7 only")
    final = COL / "stages/collection_index"
    event("WAIT_COLLECTION", required=str(COL / "receipts/collection_index.json"))
    while not receipt_ok(COL / "receipts/collection_index.json"):
        require_training_authorized()
        if (COL / "events.jsonl").exists():
            events = [json.loads(x) for x in (COL / "events.jsonl").read_text().splitlines() if x.strip()]
            if events and events[-1]["event"] in ("BLOCKED", "POOL_EXHAUSTED"):
                raise RuntimeError(f"Collection blocked: {events[-1]}")
        if not collector_alive():
            raise RuntimeError("Collection driver absent before verified completion; no automatic restart")
        time.sleep(60)
    passed(final / "result.json")
    manifest = final / "manifest.json"
    m = json.loads(manifest.read_text())
    index = json.loads((final / "collection_index.json").read_text())
    validate_manifest(m, index)

    def freeze(d):
        shutil.copy2(manifest, d / "manifest.json")
        shutil.copy2(final / "collection_index.json", d / "collection_index.json")
        dump(d / "protocol.json", {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
             "dataset_version": m["dataset_version"], "code_commit": COMMIT,
             "variant": "B", "steps": 10000, "seed": 1000, "effective_batch": 16,
             "loss_source": "historical winner recorded in docs/recovery_20261020.md; fixed for reconstruction, no new sweep",
             "omega_weight": 1, "steer_aux_weight": 0.25, "conversion_concurrency": 1,
             "encoder_queue": 4096, "gpu": 7, "original_v2_resumption": False})
    frozen = stage("manifest", freeze)
    manifest = frozen / "manifest.json"
    datasets, shards = {}, {}
    for split, n in (("train", 24), ("val", 2)):
        shards[split] = n
        converted = []
        for k in range(n):
            name = f"convert_{split}_{k:02d}"
            def convert(d, k=k, n=n, split=split):
                command(d, "convert", [P5, S / "ovmm_to_lerobot.py", "--src", final,
                        "--manifest", manifest, "--split", split, "--shard", f"{k}/{n}",
                        "--streaming", "--encoder-threads", "2", "--writer-threads", "2",
                        "--encoder-queue-maxsize", "4096", "--no-stats", "--out", d / "dataset"])
                text = (d / "convert.log").read_text(errors="replace")
                if "###CONVERT###" not in text or __import__("re").search(r"dropped.*frame", text, __import__("re").I):
                    raise RuntimeError("Missing conversion completion or dropped video frames")
                json.loads((d / "dataset/ovmm_episode_index.json").read_text())
            converted.append(stage(name, convert) / "dataset")
        def merge(d, split=split, converted=converted):
            command(d, "merge", [P5, S / "merge_shards.py", "--shards", *converted,
                                "--out", d / "dataset", "--manifest", manifest, "--split", split])
            stats = json.loads((d / "dataset/meta/exact_stats.json").read_text())
            if stats["manifest_sha256"] != hashlib.sha256(manifest.read_bytes()).hexdigest():
                raise RuntimeError("Statistics refer to another manifest")
        datasets[split] = stage(f"merge_{split}", merge) / "dataset"

    def video(d):
        command(d, "verify", [P5, S / "verify_videos.py", "--dataset",
                *[f"{datasets[sp]}:{shards[sp]}" for sp in ("train", "val")],
                "--logs", OUT / "stages/convert_*/convert.log", "--workers", "4",
                "--align-workers", "1", "--out", d / "video_verify.json"])
        passed(d / "video_verify.json")
    stage("video_verify", video)

    def config(d):
        # YAML is emitted by the installed training interpreter, not the system Python.
        command(d, "config", [P5, "-c",
            "import yaml,sys; from pathlib import Path; c=yaml.safe_load(Path(sys.argv[1]).read_text()); "
            "c.update(name='lj_reconstructed_B',init_checkpoint=sys.argv[2],train_data=sys.argv[3],val_data=sys.argv[4],"
            "out=sys.argv[5],steps=10000,seed=1000,num_workers=4,keep_ckpts=2,save_last=True,resumable=True,strict_resume=True); "
            "c['loss']['dim_weights']['base.omega']=1.0; c['loss']['steer_aux']['weight']=0.25; "
            "assert c['batch_per_gpu']*c['grad_accum']==16; "
            "assert c['history']==dict(camera='camera_top',offsets=[5,10,15],state=False); "
            "Path(sys.argv[6]).write_text(yaml.safe_dump(c,sort_keys=False))",
            CODE / "configs/alphabot2_pi05/sft_v2B.yaml", ROOT / "runtime/pi05/ckpt",
            datasets["train"], datasets["val"], MODEL, d / "sft_B.yaml"])
    cfg = stage("config", config) / "sft_B.yaml"
    def interface(d):
        command(d, "interface", [P5, S / "interface_gate.py", "--config", cfg, "--collect", final,
                                "--h-exec", "5", "--out", d / "report"], gpu=True)
        passed(d / "report/interface_gate.json")
    stage("interface", interface)

    # Stop at real optimizer boundaries, preserving the 10k LR horizon. Upload only
    # while no trainer writes the model tree. The probe's two steps count toward 10k.
    for end in (1, 2, *range(1000, 10001, 1000)):
        name = f"train_through_{end:07d}"
        def train(d, end=end):
            command(d, "sft", [P5, S / "sft_ovmm.py", "--config", cfg,
                               "--stop-after-step", end], gpu=True)
            ck = MODEL / f"ckpt_step{end:07d}"
            if not (ck / "COMPLETE").exists():
                raise RuntimeError(f"Missing complete checkpoint {ck}")
            infos = sorted(MODEL.glob("run_info_step*.json"))
            info = json.loads(infos[-1].read_text())
            if end > 1 and (not info["rng_restored"] or not info["resume_from"]):
                raise RuntimeError("Optimizer/RNG resume probe failed")
            if end == 2 and info["start_step"] != 1:
                raise RuntimeError("Resume probe did not resume step 1")
            records = [json.loads(x) for x in (MODEL / "train_log.jsonl").read_text().splitlines()]
            for row in records:
                for key in ("loss", "val_loss", "grad_norm"):
                    if key in row and not math.isfinite(row[key]):
                        raise RuntimeError(f"Non-finite training metric: {row}")
            shutil.copy2(cfg, d / "sft_B.yaml")
            shutil.copy2(infos[-1], d / "run_info.json")
            dump(d / "checkpoint.json", {"step": end, "path": str(ck), "global_batch": info["global_batch"],
                 "resume_start": info["start_step"], "rng_restored": info["rng_restored"],
                 "peak_gpu_gb": max(x.get("gpu_mem_gb", 0) for x in records),
                 "persistent_checkpoint": True, "lr_horizon": 10000})
            if end == 10000:
                if not all((MODEL / x).exists() for x in ("DONE", "best/config.json", "last/config.json")):
                    raise RuntimeError("Final/best/last checkpoint missing")
        stage(name, train, extra=(MODEL,))
        if end == 2:
            event("TRAINING_RESUME_PROBE_PASSED", steps=2, next="continue same run through step 10000")
    event("TRAINING_COMPLETE", steps=10000, model=str(MODEL), evaluation_started=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if OUT.exists():
            event("BLOCKED", error=str(e))
        raise
