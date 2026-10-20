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
* Paths inside a ".cache/huggingface/" directory (HF download-cache locks / metadata; the uploader reserves that name)
  are excluded and listed in the manifest.
* backups/<tag>/MANIFEST.json lists every uploaded file with size and sha256; after the upload the remote tree is
  listed and every file's size (and LFS sha256 where available) must match, else exit 1.
Token: $HF_TOKEN, else ~/.cache/huggingface/token, else /root/.config/hf_token_write (never printed or written).
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

from huggingface_hub import HfApi

SECRET = re.compile(rb"hf_[A-Za-z0-9]{30,}|wandb_v1_[A-Za-z0-9_]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----")
TEXT_MAX = 20 << 20
BIG = 4 << 30                                                     # files above this go one by one


def hf_call(fn, *args, **kwargs):
    """Respect server cooldown also for API methods without built-in 429 retries."""
    for attempt in range(7):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            response = getattr(exc, "response", None)
            if response is None or response.status_code != 429 or attempt == 6:
                raise
            try:
                delay = max(1.0, float(response.headers.get("Retry-After", "300"))) + 2
            except ValueError:
                delay = 302
            print(json.dumps({"event": "HF_RATE_LIMIT_WAIT", "seconds": delay, "attempt": attempt + 1}), flush=True)
            time.sleep(delay)


def token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for p in (Path(os.environ.get("HF_TOKEN_PATH", Path.home() / ".config/lmm/hf_token")),
              Path.home() / ".cache/huggingface/token", Path("/root/.config/hf_token_write")):
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
    if not src.exists() or src.is_symlink():
        raise ValueError(f"missing or symlink source: {src}")
    return [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file() and not p.is_symlink())


def verify_file(entry, size, sha256_hex, git_blob_hex):
    """LFS objects use SHA256; ordinary Git files use the blob SHA1 (not just size)."""
    if entry is None or entry.size != size:
        return False
    lfs = getattr(entry, "lfs", None)
    return (getattr(lfs, "sha256", None) == sha256_hex if lfs
            else getattr(entry, "blob_id", None) == git_blob_hex)


def git_blob(p):
    h = hashlib.sha1(f"blob {p.stat().st_size}\0".encode())
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def scan_git(g):
    # A compressed bundle cannot be checked by scanning its bytes. Inspect every reachable blob first.
    objects = subprocess.check_output(["git", "-C", g, "rev-list", "--objects", "--all"], text=True)
    for line in objects.splitlines():
        oid = line.split(" ", 1)[0]
        kind = subprocess.check_output(["git", "-C", g, "cat-file", "-t", oid], text=True).strip()
        if kind != "blob":
            continue
        size = int(subprocess.check_output(["git", "-C", g, "cat-file", "-s", oid]))
        if size <= TEXT_MAX:
            data = subprocess.check_output(["git", "-C", g, "cat-file", "blob", oid])
            if SECRET.search(data):
                sys.exit(f"refused: secret-like content in git object {oid}")


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
    ap.add_argument("--receipt", help="Write a verified receipt outside the backed-up sources.")
    ap.add_argument("items", nargs="*")
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", a.tag) or a.tag in (".", ".."):
        sys.exit("invalid backup tag")
    api = HfApi(token=token())
    hf_call(api.create_repo, a.repo, repo_type="dataset", private=True, exist_ok=True)
    if not hf_call(api.repo_info, a.repo, repo_type="dataset").private:
        sys.exit(f"refused: {a.repo} is public")
    tmp = Path(os.environ.get("HF_SYNC_STAGE", Path.home() / ".cache/hf_sync_stage")) / a.tag   # stable: re-runs resume
    if (tmp / "stage").exists():
        # Retain upload metadata but never accidentally upload stale files from an earlier invocation.
        for p in (tmp / "stage").iterdir():
            if p.name != ".cache":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    pairs = []                                                    # (local file, path in repo)
    for g in a.git:
        scan_git(g)
        sha = subprocess.check_output(["git", "-C", g, "rev-parse", "--short", "HEAD"], text=True).strip()
        b = tmp / f"{Path(g).resolve().name}_{sha}.bundle"
        subprocess.check_call(["git", "-C", g, "bundle", "create", str(b), "--all"], stderr=subprocess.DEVNULL)
        pairs.append((b, f"code/{b.name}"))
    for it in a.items:
        src, _, dest = it.partition(":")
        dest = dest or f"backups/{a.tag}/{Path(src).name}"
        if dest.startswith("/") or ".." in Path(dest).parts:
            sys.exit("invalid repository destination")
        s = Path(src)
        for p in files_of(s):
            pairs.append((p, dest if s.is_file() else f"{dest}/{p.relative_to(s).as_posix()}"))
    # huggingface_hub reserves ".cache/huggingface" for its own upload metadata and never uploads it: such files (HF
    # download caches: locks / metadata) are excluded explicitly and listed in the manifest, not silently lost
    excluded = [str(p) for p, d in pairs if "/.cache/huggingface/" in f"/{d}"]
    pairs = [(p, d) for p, d in pairs if "/.cache/huggingface/" not in f"/{d}"]
    if not pairs or len({d for _, d in pairs}) != len(pairs):
        sys.exit("empty backup or duplicate repository destinations")
    forbidden = {".env", ".netrc", "token", "stored_tokens", "hf_token", "hf_token_write", "credentials"}
    if any(p.name in forbidden or ".ssh" in p.parts for p, _ in pairs):
        sys.exit("refused: credential file in sources")
    bad = scan([p for p, _ in pairs])
    if bad:
        sys.exit(f"refused: secret-like content in {bad[:5]}")
    # Copy a stable snapshot. Hardlinks let an active writer silently mutate an in-flight backup.
    snapshot = tmp / "snapshot"
    snapshot.mkdir(exist_ok=True)
    frozen = []
    for i, (p, d) in enumerate(pairs):
        before = p.stat()
        q = snapshot / str(i)
        shutil.copy2(p, q)
        after = p.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or sha256(p) != sha256(q):
            sys.exit(f"source changed during snapshot: {p}")
        frozen.append((q, d))
    sources = [str(p) for p, _ in pairs]
    pairs = frozen
    if scan([p for p, _ in pairs]):
        sys.exit("refused: secret-like content in snapshot")
    man = [{"path": d, "size": p.stat().st_size, "sha256": sha256(p), "git_blob": git_blob(p),
            "source": sources[i]} for i, (p, d) in enumerate(pairs)]
    mf = tmp / "MANIFEST.json"
    mf.write_text(json.dumps({"tag": a.tag, "time": time.strftime("%F %T"), "host": os.uname().nodename, "files": man,
                              "excluded_hf_cache_files": excluded}, indent=1))
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
            shutil.copy2(p, q)
    if small:                                                     # resumable, split into several commits (large trees time out as one)
        hf_call(api.upload_large_folder, repo_id=a.repo, repo_type="dataset", folder_path=str(stage), print_report=False,
                num_workers=int(os.environ.get("HF_UPLOAD_WORKERS", "2")))
    for p, d in pairs:
        if p.stat().st_size >= BIG:
            hf_call(api.upload_file, path_or_fileobj=str(p), path_in_repo=d, repo_id=a.repo, repo_type="dataset", commit_message=f"backup {a.tag}: {d}")
    # verify
    want = {d: (p.stat().st_size, sha256(p), git_blob(p)) for p, d in pairs}
    revision = hf_call(api.repo_info, a.repo, repo_type="dataset").sha
    remote = {}
    # Verify only this snapshot; expanded recursive listings of all historical backups
    # use tiny pages and unnecessarily consume the shared API quota.
    paths = sorted(want)
    for start in range(0, len(paths), 100):
        for e in hf_call(api.get_paths_info, a.repo, paths[start:start + 100], repo_type="dataset", revision=revision):
            remote[e.path] = e
        if start + 100 < len(paths):
            time.sleep(1)
    bad = [d for d, (sz, sh, blob) in want.items() if not verify_file(remote.get(d), sz, sh, blob)]
    receipt = {"repo": a.repo, "revision": revision, "tag": a.tag, "files": len(pairs),
               "bytes": sum(p.stat().st_size for p, _ in pairs),
               "verified": len(pairs) - len(bad), "mismatch": bad[:10]}
    if not bad and a.receipt:
        dst = Path(a.receipt)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.with_suffix(dst.suffix + ".tmp").write_text(json.dumps(receipt, indent=2) + "\n")
        dst.with_suffix(dst.suffix + ".tmp").replace(dst)
    print(json.dumps(receipt))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
