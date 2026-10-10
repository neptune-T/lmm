"""Periodic backup of the AlphaBot 2 OVMM long run to a PRIVATE Hugging Face dataset repo.

Every INTERVAL_S: run records (gates, scheduler state/events, screens, units, qualification / snapshot /
pilot / eval results, collection results + status, interface-gate reports, SFT logs/summaries), the
runtime archive (env tarball + lockfiles, small pinned assets list). Once available: converted LeRobot
datasets and SFT 'best' checkpoints. Raw npz trajectories (TBs) stay on persistent /workspace.
The token is read from /root/.config/hf_token (or $HF_TOKEN); it is never written anywhere.

    PYTHONPATH=/workspace/runtime/pylib python3 hf_backup.py [REPO_ID]
"""
import json, os, sys, time, traceback

from huggingface_hub import HfApi

REPO = sys.argv[1] if len(sys.argv) > 1 else "Neptune-T/lmm-alphabot2-ovmm-runs"
RT = "/workspace/runtime"
INTERVAL = int(os.environ.get("INTERVAL_S", "7200"))
tok = os.environ.get("HF_TOKEN") or open("/root/.config/hf_token").read().strip()
api = HfApi(token=tok)
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
STATE = f"{RT}/runs/hf_backup_state.json"
RECORDS = ["*.jsonl", "*.json", "*.txt", "*.yaml", "*.png", "*.log"]
IGNORE = ["collect_v1/episodes/**", "**/ckpt_step*/**", "**/*.npz", "logs/**", "**/worker_logs/**"]


def up(folder, path_in_repo, allow, ignore=(), msg=""):
    if os.path.isdir(folder):
        api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=folder, path_in_repo=path_in_repo,
                          allow_patterns=list(allow), ignore_patterns=list(ignore), commit_message=msg or f"backup {path_in_repo}")


while True:
    t0 = time.time()
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"once": []}
    try:
        up(f"{RT}/runs", "runs", RECORDS, IGNORE, f"run records {time.strftime('%F %T')}")
        if "archive" not in st["once"]:
            up(f"{RT}/archive", "archive", ["*"], [], "runtime archive (env tarball + lockfiles)")
            up(f"{RT}/pi05", "pi05_assets", ["ASSETS.json"], [], "pi0.5 assets manifest")
            st["once"].append("archive")
        for base, kind, allow in ((f"{RT}/datasets", "datasets", ["**"]), (f"{RT}/sft", "sft", ["*/best/**", "*/summary.json",
                                                                                                  "*/run_info*.json", "*/train_log.jsonl"])):
            if not os.path.isdir(base):
                continue
            for name in sorted(os.listdir(base)):
                d = f"{base}/{name}"
                ready = os.path.exists(f"{d}/ovmm_episode_index.json") if kind == "datasets" else os.path.exists(f"{d}/DONE")
                key = f"{kind}/{name}"
                if ready and key not in st["once"]:
                    up(d, key, allow if kind == "datasets" else ["best/**", "summary.json", "run_info*.json", "train_log.jsonl"],
                       [], f"{kind} {name}")
                    st["once"].append(key)
        st["last_ok"] = time.strftime("%F %T")
        st["last_wall_s"] = round(time.time() - t0, 1)
        print(json.dumps({"backup_ok": st["last_ok"], "once": st["once"], "wall_s": st["last_wall_s"]}), flush=True)
    except Exception as e:
        st["last_error"] = {"time": time.strftime("%F %T"), "error": repr(e)[:300]}
        print("BACKUP_ERROR", traceback.format_exc()[-800:], flush=True)
    json.dump(st, open(STATE, "w"), indent=1)
    time.sleep(max(60, INTERVAL - (time.time() - t0)))
