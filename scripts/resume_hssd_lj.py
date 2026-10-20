"""Resume the failed HSSD transfer over HTTP, preserving verified completed cache files.

Run with the backup environment in server tmux. Does not restart the collection
driver, allocate a GPU, clear caches, or alter other jobs. No automatic retry loop.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(os.environ.get("LMM_REBUILD_ROOT", "/cluster/home/gaomingju/workspace/lmm/lmm"))
RUN = ROOT / "runtime/rebuild_20261020"
DEST = ROOT / "runtime/home-robot/data/hssd-hab"
REVISION = "4e0292b16b32c92267be1ac8a94628deccd2ef09"


def digest(p, git=False):
    h = hashlib.sha1(f"blob {p.stat().st_size}\0".encode()) if git else hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(p, obj):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(p)


def main():
    lock = (RUN / "hssd_resume.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Refuse duplicate transfer launches; inspect only this user's exact script arguments.
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = (proc / "cmdline").read_bytes().decode().split("\0")
            if any(a in ("code/scripts/fetch_ovmm_rebuild.py",
                         str(ROOT / "code/scripts/fetch_ovmm_rebuild.py")) for a in args):
                raise RuntimeError(f"Existing asset fetch PID {proc.name}; adopt it instead")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = RUN / f"hssd_http_resume_{stamp}"
    out.mkdir()
    env = {**os.environ, "HF_HOME": str(ROOT / "hf"),
           "HF_TOKEN_PATH": str(Path.home() / ".config/lmm/hf_token"),
           "HF_SYNC_STAGE": str(ROOT / "backups/stage"),
           "HF_HUB_DISABLE_XET": "1", "HF_HUB_DOWNLOAD_TIMEOUT": "120",
           "HF_HUB_ETAG_TIMEOUT": "60", "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"}
    # Verify ordinary writes, fsync, readback and cleanup on the actual destination filesystem.
    payload = os.urandom(8 << 20)
    fd, name = tempfile.mkstemp(prefix=".lmm_io_probe_", dir=DEST)
    probe = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        assert digest(probe) == hashlib.sha256(payload).hexdigest()
    finally:
        probe.unlink(missing_ok=True)
    records, repairs = [], []
    for p in sorted(DEST.rglob("*")):
        if not p.is_file() or ".cache" in p.parts:
            continue
        rel = p.relative_to(DEST).as_posix()
        meta = DEST / ".cache/huggingface/download" / (rel + ".metadata")
        sha = digest(p)
        m = meta.read_text().splitlines() if meta.exists() else []
        if len(m) < 3 or m[0] != REVISION:
            # A failed Xet transfer can leave complete content but an empty metadata file.
            # Validate against the pinned upstream object before repairing ONLY the index.
            from huggingface_hub import HfApi
            from huggingface_hub._local_folder import write_download_metadata
            from hf_sync import token
            entry = HfApi(token=token()).get_paths_info(
                "hssd/hssd-hab", [rel], repo_type="dataset", revision=REVISION)[0]
            expected = entry.lfs.sha256 if entry.lfs else entry.blob_id
            actual = sha if entry.lfs else digest(p, git=True)
            assert p.stat().st_size == entry.size and actual == expected, f"Damaged cached content: {rel}"
            before = p.stat().st_mtime_ns
            repairs.append({"path": rel, "old_metadata": m, "revision": REVISION,
                            "upstream_etag": expected, "size": entry.size, "sha256": sha})
            write_download_metadata(DEST, rel, REVISION, expected)
            assert p.stat().st_mtime_ns == before and digest(p) == sha
            write(out / "cache_metadata_repairs.json", repairs)
            m = meta.read_text().splitlines()
        assert m[0] == REVISION, f"Unexpected cached revision: {rel}"
        etag = sha if len(m[1]) == 64 else digest(p, git=True)
        assert etag == m[1], f"Cached file hash mismatch: {rel}"
        records.append({"path": rel, "size": p.stat().st_size, "sha256": sha,
                        "mtime_ns": p.stat().st_mtime_ns})
    write(out / "cache_before.json", {"revision": REVISION, "files": records})
    write(out / "diagnosis.json", {
        "old_log": str(ROOT / "logs/ovmm_assets.log"),
        "failure": "Xet file reconstruction: Communication error on send (os error 70)",
        "root_cause_limit": "Underlying transport/filesystem cause not proven by traceback",
        "destination_write_fsync_readback": "passed 8 MiB",
        "cached_files_verified": len(records), "cached_bytes": sum(r["size"] for r in records),
        "cache_metadata_repairs": len(repairs),
        "partial_files": len(list((DEST / ".cache").rglob("*.incomplete"))),
        "repair": "HF_HUB_DISABLE_XET=1; pinned revision; force_download remains false",
        "resume_scope": "completed-file cache reuse; hub 1.19.0 removes failed temporary files",
        "existing_driver": "left running; resumes only after scenes_backup_verified.json",
    })

    def backup(suffix):
        subprocess.run([sys.executable, str(ROOT / "code/scripts/hf_sync.py"),
                        "--tag", f"lj_hssd_http_{stamp}_{suffix}", str(out),
                        "--receipt", str(RUN / f"hssd_http_{stamp}_{suffix}_backup_verified.json")],
                       env=env, check=True)

    print(json.dumps({"event": "HSSD_HTTP_CACHE_VERIFIED", "files": len(records),
                      "bytes": sum(r["size"] for r in records), "diagnostics": str(out)}), flush=True)
    backup("before")
    rc = subprocess.call([sys.executable, str(ROOT / "code/scripts/fetch_ovmm_rebuild.py")],
                         cwd=ROOT, env=env)
    changed = [r["path"] for r in records
               if not (DEST / r["path"]).is_file()
               or (DEST / r["path"]).stat().st_mtime_ns != r["mtime_ns"]
               or digest(DEST / r["path"]) != r["sha256"]]
    write(out / "resume_result.json", {"exit_code": rc, "previous_files_unchanged": not changed,
                                     "changed_files": changed, "checked_cached_files": len(records)})
    backup("after")
    if changed:
        raise RuntimeError("Previously completed cached files changed; inspect resume_result.json")
    return rc


if __name__ == "__main__":
    sys.exit(main())
