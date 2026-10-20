"""Pack immutable, upstream-verified HSSD files into self-contained HF backup volumes.

Preserves all prior remote files and local uploader metadata. No GPU usage.
The scenes receipt is published only after archive members AND remote hashes pass.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import fcntl

from hf_sync import sha256

ROOT = Path(os.environ.get("LMM_REBUILD_ROOT", "/cluster/home/gaomingju/workspace/lmm/lmm"))
RUN = ROOT / "runtime/rebuild_20261020"
SRC = ROOT / "runtime/home-robot/data/hssd-hab"
OUT = RUN / "hssd_archive_v1"
REV = "4e0292b16b32c92267be1ac8a94628deccd2ef09"


def write(p, obj):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(p)


def groups(records, limit=2 * 2**30):
    chunk, size = [], 0
    for r in records:
        if chunk and size + r["size"] > limit:
            yield chunk
            chunk, size = [], 0
        chunk.append(r)
        size += r["size"]
    if chunk:
        yield chunk


def verify_tar(path, records):
    expected = {r["path"]: r for r in records}
    seen = set()
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if (not member.isfile() or member.name not in expected or member.name in seen
                    or Path(member.name).is_absolute() or ".." in Path(member.name).parts):
                raise RuntimeError(f"Unexpected archive member: {member.name}")
            r = expected[member.name]
            h = hashlib.sha256()
            with archive.extractfile(member) as f:
                for b in iter(lambda: f.read(4 << 20), b""):
                    h.update(b)
            if member.size != r["size"] or h.hexdigest() != r["sha256"]:
                raise RuntimeError(f"Archive content mismatch: {member.name}")
            seen.add(member.name)
    if seen != set(expected):
        raise RuntimeError("Archive missing files")


def main():
    OUT.mkdir(exist_ok=True)
    lock = (RUN / "hssd_archive.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((RUN / "scenes_upstream_manifest.json").read_text())
    records = manifest["files"]
    assert manifest["revision"] == REV
    assert len(records) == 56669 and sum(r["size"] for r in records) == 10480470537
    assert len({r["path"] for r in records}) == len(records)
    volumes = []
    for k, batch in enumerate(groups(records)):
        path = OUT / f"hssd-{k:03d}.tar"
        if not path.exists():
            tmp = path.with_suffix(".tar.partial")
            if tmp.exists():
                raise RuntimeError(f"Preserve incomplete archive for diagnosis: {tmp}")
            with tarfile.open(tmp, "w:") as archive:
                for r in batch:
                    rel = Path(r["path"])
                    if rel.is_absolute() or ".." in rel.parts:
                        raise RuntimeError(f"Unsafe source path: {rel}")
                    p = SRC / rel
                    if p.is_symlink() or not p.is_file() or p.stat().st_size != r["size"]:
                        raise RuntimeError(f"Missing/changed source: {p}")
                    archive.add(p, arcname=r["path"], recursive=False)
            verify_tar(tmp, batch)
            tmp.replace(path)
        else:
            verify_tar(path, batch)
        volumes.append({"archive": path.name, "files": len(batch), "bytes": path.stat().st_size,
                        "sha256": sha256(path), "members": batch})
        print(json.dumps({"event": "ARCHIVE_VERIFIED", **{k:v for k,v in volumes[-1].items() if k != "members"}}), flush=True)
    write(OUT / "archive_manifest.json", {"upstream_repo": manifest["repo"], "upstream_revision": REV,
          "files": len(records), "bytes": sum(r["size"] for r in records), "volumes": volumes,
          "prior_remote_backup_preserved": "backups/lj_20261020_ovmm_scenes/",
          "restore": "Verify volume SHA256, extract all .tar volumes into hssd-hab, then verify every member against this manifest."})
    (OUT / "RESTORE.txt").write_text(
        "Complete HSSD backup at upstream revision " + REV + "\n"
        "Download all volumes and archive_manifest.json from this backup revision.\n"
        "Check each archive SHA256. Extract all tar volumes into a fresh hssd-hab directory.\n"
        "Verify every extracted file size and SHA256 against archive_manifest.json.\n"
        "Prior loose-file uploads are preserved but not required to restore this complete archive.\n")
    env = {**os.environ, "HF_UPLOAD_WORKERS": "2", "HF_HOME": str(ROOT / "hf"),
           "HF_TOKEN_PATH": str(Path.home() / ".config/lmm/hf_token"),
           "HF_SYNC_STAGE": str(ROOT / "backups/stage"), "CUDA_VISIBLE_DEVICES": ""}
    receipt = RUN / "scenes_archive_backup_verified.json"
    subprocess.run([str(ROOT / "envs/backup/bin/python"), str(ROOT / "code/scripts/hf_sync.py"),
                    "--tag", "lj_20261020_hssd_archive_v1", str(OUT),
                    str(RUN / "scenes_upstream_manifest.json"), "--receipt", str(receipt)],
                   env=env, check=True)
    proof = json.loads(receipt.read_text())
    assert proof["verified"] == proof["files"] and proof["mismatch"] == [] and proof["revision"]
    proof.update(format="verified_tar_volumes", asset_files=len(records), upstream_revision=REV,
                 archive_manifest_sha256=sha256(OUT / "archive_manifest.json"),
                 note="files/verified count remote backup objects; asset_files counts fully hash-verified archive members")
    final = RUN / "scenes_backup_verified.json"
    if final.exists():
        raise RuntimeError("Another scenes receipt already exists; preserve and inspect both")
    write(final, proof)
    print(json.dumps({"event": "SCENES_ARCHIVE_BACKUP_VERIFIED", **proof}), flush=True)


if __name__ == "__main__":
    main()
