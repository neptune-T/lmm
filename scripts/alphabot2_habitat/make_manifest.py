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
                             [--train-from-index DS/ovmm_episode_index.json] [--supervisor-log LOG] [--legacy-commit C]

Provenance of rows written before the collector recorded code_commit: the row's worker id -> that worker's
latest START_WORKER (supervisor log) before the row was written -> the repo HEAD at that moment (git reflog
of the deployed checkout). Only if that fails, --legacy-commit.
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


def worker_starts(log):
    """{worker id: [start epochs]} from the collection supervisor log."""
    out = {}
    if not log or not os.path.exists(log):
        return out
    for l in open(log, errors="replace"):
        p = l.split()
        if len(p) >= 4 and p[2] == "START_WORKER":
            try:
                t = time.mktime(time.strptime(f"{p[0]} {p[1]}", "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                continue
            out.setdefault(int(p[3]), []).append(t)
    return out


def head_history():
    """[(epoch, short commit)] of the deployed checkout's HEAD moves, oldest first."""
    h = []
    for l in git("log", "-g", "--date=unix", "--format=%h %gd").splitlines():
        c, _, gd = l.partition(" ")
        try:
            h.append((float(gd[gd.index("{") + 1:gd.index("}")]), c))
        except ValueError:
            continue
    return sorted(h)


def commit_at(hist, t):
    c = None
    for ht, hc in hist:
        if ht <= t:
            c = hc
    return c


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
    ap.add_argument("--legacy-commit", default="", help="last-resort collector commit for rows without code_commit")
    ap.add_argument("--supervisor-log", default=None, help="collection supervisor log (worker start times)")
    ap.add_argument("--extra-start", action="append", default=[],
                    help="'W1,W2@YYYY-mm-dd HH:MM:SS' start of workers not launched by the supervisor (e.g. the clean pilot)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = accepted(a.src)
    starts, hist = worker_starts(a.supervisor_log), head_history()
    for x in a.extra_start:
        ws, _, ts = x.partition("@")
        for w in ws.split(","):
            starts.setdefault(int(w), []).append(time.mktime(time.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S")))

    def provenance(r, mt):
        if r.get("code_commit"):
            return r["code_commit"], "row"
        ts = [t for t in starts.get(int(r.get("worker", -1)), []) if mt is None or t <= mt]
        if ts and hist and commit_at(hist, max(ts)):
            return commit_at(hist, max(ts)), f"worker {r.get('worker')} started {time.strftime('%F %T', time.localtime(max(ts)))}"
        return a.legacy_commit, "legacy"
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
        cc, how = provenance(r, snap[u][1] if u in snap else None)
        return {"unit": u, "episode": str(r.get("episode") or r["episode_id"]), "replica": r.get("replica"),
                "arm": r.get("arm"), "scene": r.get("scene"), "dir": r["dir"], "worker": r.get("worker"),
                "code_commit": cc, "code_commit_from": how,
                "accepted_at": time.strftime("%F %T", time.localtime(snap[u][1])) if u in snap else None}

    details = {u: info(u) for u in train + val}
    commits = sorted({d["code_commit"] for d in details.values()})
    m = {"round": a.round, "created": time.strftime("%F %T"), "src": a.src,
         "snapshot_cutoff": time.strftime("%F %T", time.localtime(a.cutoff)), "snapshot_cutoff_epoch": a.cutoff,
         "snapshot_accepted": len(snap), "accepted_total_at_creation": len(rows),
         "split_rule": f"crc32('sft-split:<episode>') % 1000 < {a.val_permille} -> val (whole episodes)",
         "train_rule": f"train-split snapshot sorted by unit id, first {a.max_train or 'all'}",
         "train": train, "val": val, "n_train": len(train), "n_val": len(val),
         "train_sha256": sha(train), "val_sha256": sha(val),
         "train_detail": [details[u] for u in train], "val_detail": [details[u] for u in val],
         "provenance_sources": sorted({d["code_commit_from"].split()[0] for d in details.values()}),
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
