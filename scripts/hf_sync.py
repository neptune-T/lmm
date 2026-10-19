"""Task-completion backup to the PRIVATE Hugging Face dataset repo (rule: after every data collection, training run,
evaluation result and code change, run this; a task is backed up only when this exits 0).

    python scripts/hf_sync.py --tag 2026-10-19_pa_og_results [--repo Neptune-T/lmm-alphabot2-ovmm-runs] \
        [--git /path/to/repo] SRC[:DEST] [SRC[:DEST] ...]

* SRC is a file or a directory; DEST (path in the repo) defaults to backups/<tag>/<basename of SRC>.
* --git REPO_DIR uploads `git bundle --all` of that repository as code/<name>_<short sha>.bundle (restore with
  `git clone <bundle>`), so code is backed up even when GitHub is unreachable.
* The repo is created PRIVATE if missing; a public repo is refused.
* Text files are scanned for secrets (HF / W&B tokens, private keys) and the upload is refused if any match.
* Upload: hf_hub upload_large_folder (resumable, multi-commit; re-running the same command resumes) for files < 4 GB,
  upload_file for larger ones.
* backups/<tag>/MANIFEST.json lists every uploaded file with size and sha256; after the upload the remote tree is
  listed and every file's size (and LFS sha256 where available) must match, else exit 1.
Token: $HF_TOKEN, else ~/.cache/huggingface/token, else /root/.config/hf_token_write (never printed or written).
"""
import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

from huggingface_hub import HfApi

SECRET = re.compile(rb"hf_[A-Za-z0-9]{30,}|wandb_v1_[A-Za-z0-9_]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----")
TEXT_MAX = 20 << 20
BIG = 4 << 30                                                     # files above this go one by one


def token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for p in (Path.home() / ".cache/huggingface/token", Path("/root/.config/hf_token_write")):
        if p.exists():
            return p.read_text().strip()
    sys.exit("no HF token (HF_TOKEN, ~/.cache/huggingface/token or /root/.config/hf_token_write)")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def files_of(src):
    src = Path(src)
    return [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file() and not p.is_symlink())


def scan(paths):
    bad = []
    for p in paths:
        if p.stat().st_size <= TEXT_MAX:
            b = p.read_bytes()
            if b"\0" not in b[:8192] and SECRET.search(b):
                bad.append(str(p))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Neptune-T/lmm-alphabot2-ovmm-runs")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--git", action="append", default=[])
    ap.add_argument("items", nargs="*")
    a = ap.parse_args()
    api = HfApi(token=token())
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(a.repo, repo_type="dataset").private:
        sys.exit(f"refused: {a.repo} is public")
    tmp = Path(os.environ.get("HF_SYNC_STAGE", Path.home() / ".cache/hf_sync_stage")) / a.tag   # stable: re-runs resume
    if (tmp / "stage").exists():
        import shutil
        shutil.rmtree(tmp / "stage" / "backups", ignore_errors=True)   # re-link current files; upload cache is kept
    tmp.mkdir(parents=True, exist_ok=True)
    pairs = []                                                    # (local file, path in repo)
    for g in a.git:
        sha = subprocess.check_output(["git", "-C", g, "rev-parse", "--short", "HEAD"], text=True).strip()
        b = tmp / f"{Path(g).resolve().name}_{sha}.bundle"
        subprocess.check_call(["git", "-C", g, "bundle", "create", str(b), "--all"], stderr=subprocess.DEVNULL)
        pairs.append((b, f"code/{b.name}"))
    for it in a.items:
        src, _, dest = it.partition(":")
        dest = dest or f"backups/{a.tag}/{Path(src).name}"
        s = Path(src)
        for p in files_of(s):
            pairs.append((p, dest if s.is_file() else f"{dest}/{p.relative_to(s).as_posix()}"))
    bad = scan([p for p, _ in pairs])
    if bad:
        sys.exit(f"refused: secret-like content in {bad[:5]}")
    man = [{"path": d, "size": p.stat().st_size, "sha256": sha256(p), "source": str(p)} for p, d in pairs]
    mf = tmp / "MANIFEST.json"
    mf.write_text(json.dumps({"tag": a.tag, "time": time.strftime("%F %T"), "host": os.uname().nodename, "files": man}, indent=1))
    pairs.append((mf, f"backups/{a.tag}/MANIFEST.json"))
    small = [(p, d) for p, d in pairs if p.stat().st_size < BIG]
    stage = tmp / "stage"                                         # one commit for all small files (hardlinks/copies)
    for p, d in small:
        q = stage / d; q.parent.mkdir(parents=True, exist_ok=True)
        if q.exists():
            q.unlink()
        try:
            os.link(p, q)
        except OSError:
            import shutil
            shutil.copy2(p, q)
    if small:                                                     # resumable, split into several commits (large trees time out as one)
        api.upload_large_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(stage), print_report=False)
    for p, d in pairs:
        if p.stat().st_size >= BIG:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=d, repo_id=a.repo, repo_type="dataset", commit_message=f"backup {a.tag}: {d}")
    # verify
    shas = {m["path"]: m["sha256"] for m in man}
    want = {d: (p.stat().st_size, shas.get(d)) for p, d in pairs}
    remote = {}
    for top in sorted({d.split("/")[0] for d in want}):
        for e in api.list_repo_tree(a.repo, repo_type="dataset", path_in_repo=top, recursive=True, expand=True):
            if hasattr(e, "size") and e.size is not None and not hasattr(e, "tree_id"):
                remote[e.path] = (e.size, getattr(e.lfs, "sha256", None) if getattr(e, "lfs", None) else None)
    bad = [d for d, (sz, sh) in want.items() if d not in remote or remote[d][0] != sz or (sh and remote[d][1] and remote[d][1] != sh)]
    print(json.dumps({"repo": a.repo, "tag": a.tag, "files": len(pairs), "bytes": sum(p.stat().st_size for p, _ in pairs),
                      "verified": len(pairs) - len(bad), "mismatch": bad[:10]}))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
