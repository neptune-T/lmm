"""Stage the OVMM data tree from the bosfs archive into a fast runtime tree (/dev/shm).

bosfs costs ~20-30 ms per small-file open even warm, and Habitat's sim init reads every HSSD
object config (~16.5k) plus scene/stage files. Small files are therefore copied; the 35k HSSD
object meshes (objects/**/*.glb, 10.2 GB, read once per scene load) are symlinked to the archive.
OVMM objects and episodes are copied whole (~1.7 GB).

    python stage_data.py SRC_DATA_DIR DST_DATA_DIR      (idempotent; DST/.staged marks completion)
"""
import os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor

src, dst = sys.argv[1], sys.argv[2]
t0 = time.time()
jobs = []                                            # (kind, rel)
for rel_root, man in (("hssd-hab", "MANIFEST.4e0292b.tsv"), ("objects", "MANIFEST.e9b714c.tsv")):
    for line in open(os.path.join(src, rel_root, man)):
        p = line.split("\t", 1)[0]
        big = rel_root == "hssd-hab" and p.startswith("objects/") and p.endswith(".glb")
        jobs.append(("link" if big else "copy", f"{rel_root}/{p}"))
for rel_root in ("datasets/ovmm", "robots/hab_stretch"):
    for r, _, fs in os.walk(os.path.join(src, rel_root)):
        for f in fs:
            jobs.append(("copy", os.path.relpath(os.path.join(r, f), src)))
for f in os.listdir(src):
    if os.path.isfile(os.path.join(src, f)):
        jobs.append(("copy", f))
for d in sorted({os.path.dirname(os.path.join(dst, rel)) for _, rel in jobs}):
    os.makedirs(d, exist_ok=True)


def do(job):
    kind, rel = job
    s, d = os.path.join(src, rel), os.path.join(dst, rel)
    if os.path.lexists(d):
        return kind, 0
    if kind == "link":
        os.symlink(s, d)
        return kind, 0
    shutil.copyfile(s, d)
    return kind, os.path.getsize(d)


n = {"copy": 0, "link": 0}
nbytes = 0
with ThreadPoolExecutor(32) as ex:
    for kind, b in ex.map(do, jobs, chunksize=64):
        n[kind] += 1
        nbytes += b
open(os.path.join(dst, ".staged"), "w").write(f"{n}\n")
print("STAGED", dst, n, f"{nbytes / 1e9:.2f} GB copied", f"{time.time() - t0:.0f}s", flush=True)
