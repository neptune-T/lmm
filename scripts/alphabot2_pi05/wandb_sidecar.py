#!/usr/bin/env python3
"""W&B mirror of one v2 SFT run, as a sidecar: it only READS the run's files and never touches training.

Tails RUN/train_log.jsonl (train + val records) and logs them with trainer/step as the x-axis; adds the
training GPU's utilisation / memory, checkpoint saves (COMPLETE) and persistent mirrors (.MIRRORED), and the
driver's sft_exit events; after DONE, summary.json; after the driver's V2_DONE, the variant's block of
v2_comparison.json as eval/* summary values. Idempotent: the number of consumed log lines / events is kept in
STATE, and the W&B run id is fixed, so a restart resumes the same run without duplicates.
The API key is read from KEY_FILE (mode 600, configured by the user) and never printed; until it exists the
sidecar waits.
  wandb_sidecar.py --variant A --gpu 0 --code-commit 3b08dad
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True)
ap.add_argument("--gpu", required=True)
ap.add_argument("--code-commit", required=True, help="commit the training was launched from")
ap.add_argument("--tag", default="v2")
ap.add_argument("--runs", default="/workspace/runtime/runs")
ap.add_argument("--sft", default="/dev/shm/lmm_runtime/sft")
ap.add_argument("--mirror", default="/workspace/runtime/sft_mirror")
ap.add_argument("--key-file", default="/root/.config/wandb_api_key")
ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "2980976495-peking-university"))
ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "VLA-baseline"))
ap.add_argument("--poll", type=float, default=60)
a = ap.parse_args()

V, TAG = a.variant, a.tag
RUN, MIR, ROOT = Path(a.sft) / f"{TAG}{V}", Path(a.mirror) / f"{TAG}{V}", Path(a.runs) / TAG
STATE = ROOT / f"wandb_{V}_state.json"
say = lambda *m: print(time.strftime("%F %T"), f"[wandb {V}]", *m, flush=True)

while not (Path(a.key_file).is_file() and Path(a.key_file).read_text().strip()):
    say(f"waiting for {a.key_file}"); time.sleep(300)
while not (RUN / "train_log.jsonl").is_file():
    say("waiting for train_log.jsonl"); time.sleep(a.poll)

os.environ["WANDB_API_KEY"] = Path(a.key_file).read_text().strip()
os.environ.setdefault("WANDB_DIR", f"/dev/shm/wandb_{TAG}")
os.environ.setdefault("WANDB_SILENT", "true")
os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
import wandb

M = json.load(open(Path(a.runs) / "manifests" / f"{TAG}.json"))
info = sorted(RUN.glob("run_info_step*.json"))
cfg = {"variant": V, "code_commit": a.code_commit, "dataset_version": M.get("dataset_version"),
       "manifest_commit": M.get("manifest_commit"), "n_train": M.get("n_train"), "n_val": M.get("n_val"),
       "train_sha256": M.get("train_sha256"), "val_sha256": M.get("val_sha256"), "train_gpu": a.gpu}
if info:
    cfg["run_info"] = json.load(open(info[0]))
run = wandb.init(entity=a.entity, project=a.project, name=f"AlphaBot2-OVMM-{TAG}-{V}",
                 id=f"alphabot2-ovmm-{TAG}-{V.lower()}-{M.get('dataset_version')}", resume="allow",
                 group=f"AlphaBot2-OVMM-{TAG}-{M.get('dataset_version')}", job_type="sft",
                 tags=[TAG, f"variant-{V}", M.get("dataset_version") or "", a.code_commit], config=cfg)
run.define_metric("trainer/step")
run.define_metric("*", step_metric="trainer/step")

st = json.load(open(STATE)) if STATE.is_file() else {"lines": 0, "events": 0, "ckpt": [], "mirrored": [], "done": False}
save = lambda: STATE.write_text(json.dumps(st))      # small whole-file rewrite (no renames on bosfs)
last_step = 0


def flat(d, pre):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(flat(v, f"{pre}{k}/"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{pre}{k}"] = v
        elif isinstance(v, str) and "/" in v and v.replace("/", "").isdigit():   # "3/30" -> count and rate
            n, d_ = map(int, v.split("/"))
            out[f"{pre}{k}_count"], out[f"{pre}{k}_n"] = n, d_
            out[f"{pre}{k}_rate"] = n / d_ if d_ else 0.0
    return out


while True:
    lines = (RUN / "train_log.jsonl").read_text().splitlines() if (RUN / "train_log.jsonl").is_file() else []
    for l in lines[st["lines"]:]:
        try:
            r = json.loads(l)
        except ValueError:
            break                                                  # partially written line: retry next poll
        last_step = r.get("step", last_step)
        if "val_loss" in r:
            rec = {"val/loss": r["val_loss"], **flat(r.get("val_group"), "val/group/"), "val/saved_best": int(bool(r.get("saved_best")))}
        else:
            rec = {f"train/{k}": v for k, v in r.items() if k != "step" and isinstance(v, (int, float))}
            rec.update(flat(r.get("group_loss"), "train/group/"))
        run.log({"trainer/step": last_step, **rec})
        st["lines"] += 1
    try:
        q = subprocess.run(["nvidia-smi", "-i", str(a.gpu), "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout.split(",")
        run.log({"trainer/step": last_step, "sys/gpu_util_pct": float(q[0]), "sys/gpu_mem_used_gb": float(q[1]) / 1024})
    except Exception:
        pass
    for c in sorted(RUN.glob("ckpt_step*/COMPLETE")):
        s = int(c.parent.name.replace("ckpt_step", ""))
        if s not in st["ckpt"]:
            st["ckpt"].append(s); run.log({"trainer/step": last_step, "ckpt/saved_step": s})
    for c in sorted(MIR.glob("ckpt_step*/.MIRRORED")):
        s = int(c.parent.name.replace("ckpt_step", ""))
        if s not in st["mirrored"]:
            st["mirrored"].append(s); run.log({"trainer/step": last_step, "ckpt/mirrored_step": s})
    ev = (ROOT / "events.jsonl").read_text().splitlines() if (ROOT / "events.jsonl").is_file() else []
    v2_done = False
    for l in ev[st["events"]:]:
        try:
            e = json.loads(l)
        except ValueError:
            break
        st["events"] += 1
        if e.get("event") == "sft_exit" and e.get("variant") == V:
            run.log({"trainer/step": last_step, "driver/sft_exit_rc": e.get("rc"), "driver/sft_attempt": e.get("attempt")})
    v2_done = any('"V2_DONE"' in l for l in ev)
    if (RUN / "DONE").is_file() and not st["done"]:
        run.summary.update(flat(json.load(open(RUN / "summary.json")), "final/"))
        st["done"] = True
    save()
    if v2_done and (ROOT / "v2_comparison.json").is_file():
        cmp_ = json.load(open(ROOT / "v2_comparison.json"))
        run.summary.update(flat(cmp_.get(f"{TAG}{V}"), "eval/"))
        run.summary.update(flat(cmp_.get("sanity_v1(fp32)"), "eval_sanity_v1/"))
        say("final eval logged; finishing")
        run.finish()
        sys.exit(0)
    time.sleep(a.poll)
