"""Frozen SFT-round manifest: the exact accepted trajectories a round trains / validates on (stdlib only).

The collection keeps growing while a round converts and trains, so a round NEVER scans it: this manifest
fixes the sample IDs once, at a snapshot cutoff, and the converter reads exactly these IDs.

  snapshot     accepted rows (OVMM success + fresh-reset replay pass) whose episode meta.json (written last,
               before the results row) is no newer than the cutoff
  split        crc32("sft-split:<episode>") % 1000 < val_permille -> val (whole episodes; = ovmm_to_lerobot)
  train        the train-split snapshot sorted by unit id, capped at --max-train (0 = all)
  val          every val-split snapshot row (SFT validation loss only; never used for evaluation)

--train-from-index adopts a train list that a conversion already produced (its ovmm_episode_index.json):
the manifest then records that exact list and verifies it against the snapshot rule (mismatches listed).

    python3 make_manifest.py --src RUN --round sanity_v1 --cutoff EPOCH --max-train 200 --out M.json
                             [--train-from-index DS/ovmm_episode_index.json] [--legacy-commit 7e62b2e]
"""
import argparse, glob, hashlib, json, os, subprocess, time, zlib

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
PATHS = {"expert": "lmm/embodiments/alphabot2/habitat/expert.py",
         "reset_contract": "lmm/embodiments/alphabot2/habitat/env.py",
         "policy_interface": "lmm/embodiments/alphabot2/ovmm_policy_io.py"}


def split_of(episode, val_permille):
    return "val" if zlib.crc32(f"sft-split:{episode}".encode()) % 1000 < val_permille else "train"


def accepted(src):
    names = set(glob.glob(f"{src}/results*.jsonl")) | {f"{src}/results_w{i:02d}.jsonl" for i in range(100)}
    rows = {}
    for f in sorted(n for n in names if os.path.exists(n)):     # bosfs readdir lags: probe worker files too
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("accepted") and r.get("dir"):
                rows[str(r["episode_id"])] = r
    return rows


def git(*a):
    try:
        return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def last_change(commit, path):
    """The last commit at or before `commit` that changed `path` (the version the data was produced with)."""
    return git("log", "-1", "--format=%h", commit, "--", path) if commit else ""


def sha(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--round", required=True)
    ap.add_argument("--cutoff", type=float, required=True, help="snapshot cutoff (epoch seconds)")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--val-permille", type=int, default=50)
    ap.add_argument("--train-from-index", default=None)
    ap.add_argument("--legacy-commit", default="", help="collector commit for rows without code_commit")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = accepted(a.src)
    snap = {}
    for u, r in rows.items():
        try:
            mt = os.path.getmtime(os.path.join(r["dir"], "meta.json"))
        except OSError:
            continue
        if mt <= a.cutoff:
            snap[u] = (r, mt)
    by = {"train": [], "val": []}
    for u in sorted(snap):
        r = snap[u][0]
        by[split_of(r.get("episode") or r["episode_id"], a.val_permille)].append(u)
    rule_train = by["train"][: a.max_train] if a.max_train else by["train"]
    verification = None
    if a.train_from_index:
        idx = json.load(open(a.train_from_index))
        train = [str(e["unit"]) for e in idx["episodes"]]
        verification = {"adopted_from": a.train_from_index, "matches_snapshot_rule": train == rule_train,
                        "in_index_not_rule": sorted(set(train) - set(rule_train)),
                        "in_rule_not_index": sorted(set(rule_train) - set(train))}
        missing = [u for u in train if u not in rows]
        if missing:
            raise SystemExit(f"adopted train ids not accepted in the collection: {missing[:10]}")
    else:
        train = rule_train
    val = by["val"]
    if set(train) & set(val):
        raise SystemExit("train/val overlap")

    def info(u):
        r = rows[u]
        return {"unit": u, "episode": str(r.get("episode") or r["episode_id"]), "replica": r.get("replica"),
                "arm": r.get("arm"), "scene": r.get("scene"), "dir": r["dir"],
                "code_commit": r.get("code_commit") or a.legacy_commit,
                "accepted_at": time.strftime("%F %T", time.localtime(snap[u][1])) if u in snap else None}

    commits = sorted({info(u)["code_commit"] for u in train + val})
    m = {"round": a.round, "created": time.strftime("%F %T"), "src": a.src,
         "snapshot_cutoff": time.strftime("%F %T", time.localtime(a.cutoff)), "snapshot_cutoff_epoch": a.cutoff,
         "snapshot_accepted": len(snap), "accepted_total_at_creation": len(rows),
         "split_rule": f"crc32('sft-split:<episode>') % 1000 < {a.val_permille} -> val (whole episodes)",
         "train_rule": f"train-split snapshot sorted by unit id, first {a.max_train or 'all'}",
         "train": train, "val": val, "n_train": len(train), "n_val": len(val),
         "train_sha256": sha(train), "val_sha256": sha(val),
         "train_detail": [info(u) for u in train], "val_detail": [info(u) for u in val],
         "train_verification": verification,
         "collector_commits": commits,
         "code_versions": {k: {c: last_change(c, p) for c in commits} for k, p in PATHS.items()},
         "reset_contract": "hard world rebuild -> seeded task reset -> episode (frozen)",
         "manifest_commit": git("rev-parse", "--short", "HEAD")}
    m["dataset_version"] = f"{a.round}-" + sha([m["train_sha256"], m["val_sha256"]])[:12]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(m, open(a.out + ".tmp", "w"), indent=1)
    os.replace(a.out + ".tmp", a.out)
    print("###MANIFEST###" + json.dumps({k: m[k] for k in ("round", "snapshot_cutoff", "n_train", "n_val", "dataset_version",
                                                            "train_verification", "collector_commits", "code_versions")}))


if __name__ == "__main__":
    main()
