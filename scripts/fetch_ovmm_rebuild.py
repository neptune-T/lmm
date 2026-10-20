"""Restore the three original upstream OVMM asset revisions; verify and back up each completed dataset."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from hf_sync import sha256, token

ROOT = Path(os.environ.get("LMM_REBUILD_ROOT", "/cluster/home/gaomingju/workspace/lmm/lmm"))
RUN = ROOT / "runtime/rebuild_20261020"
ASSETS = [
    ("ai-habitat/OVMM_episodes", "9ad25fbd86a3fd352c7a0fc1f99132fbb5802378", "datasets/ovmm", "episodes"),
    ("ai-habitat/OVMM_objects", "e9b714c37ebcc55f4702a8199fd92d6b5432553a", "objects", "objects"),
    ("hssd/hssd-hab", "4e0292b16b32c92267be1ac8a94628deccd2ef09", "hssd-hab", "scenes"),
]


def main():
    api = HfApi(token=token())
    RUN.mkdir(parents=True, exist_ok=True)
    for repo, revision, rel, name in ASSETS:
        receipt = RUN / f"{name}_backup_verified.json"
        if receipt.exists():
            print(f"SKIP_VERIFIED_BACKUP {name}", flush=True)
            continue
        dest = ROOT / "runtime/home-robot/data" / rel
        files = [e for e in api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True)
                 if hasattr(e, "size")]
        print(json.dumps({"event": "DOWNLOAD", "repo": repo, "revision": revision,
                          "files": len(files), "bytes": sum(e.size for e in files)}), flush=True)
        snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=dest,
                          token=token(), max_workers=4)
        records = []
        for e in files:
            p = dest / e.path
            if not p.is_file() or p.stat().st_size != e.size:
                raise RuntimeError(f"missing/size mismatch {repo}: {e.path}")
            digest = sha256(p)
            if e.lfs:
                ok = digest == e.lfs.sha256
            else:
                h = hashlib.sha1(f"blob {e.size}\0".encode())
                with p.open("rb") as f:
                    for b in iter(lambda: f.read(1 << 22), b""):
                        h.update(b)
                ok = h.hexdigest() == e.blob_id
            if not ok:
                raise RuntimeError(f"hash mismatch {repo}: {e.path}")
            records.append({"path": e.path, "size": e.size, "sha256": digest})
        manifest = RUN / f"{name}_upstream_manifest.json"
        manifest.write_text(json.dumps({"repo": repo, "revision": revision, "files": records}, indent=1))
        subprocess.run([sys.executable, str(Path(__file__).with_name("hf_sync.py")),
                        "--tag", f"lj_20261020_ovmm_{name}", str(dest), str(manifest),
                        "--receipt", str(receipt)], check=True)
        print(f"DATASET_BACKUP_VERIFIED {name}", flush=True)


if __name__ == "__main__":
    main()
