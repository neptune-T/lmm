"""Fetch the pi0.5 SFT assets onto the persistent archive (no env needed; system python + pylib):
  * the public AlphaBot pi0.5 checkpoint alphabot2/aibot2_grab_bag_combined_over_under_pi05 (HF, anonymous),
    pinned to the revision resolved at fetch time (recorded in ASSETS.json);
  * openpi's public PaliGemma SentencePiece tokenizer (the HF PaliGemma repo is gated).
Per-file download with size (+ LFS sha256) verification; existing correct files are skipped.

    PYTHONPATH=/workspace/runtime/pylib python3 fetch_assets.py /workspace/runtime/pi05
"""
import hashlib, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi

REPO = "alphabot2/aibot2_grab_bag_combined_over_under_pi05"
SPM_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
dest = sys.argv[1]
ck = os.path.join(dest, "ckpt")
os.makedirs(ck, exist_ok=True)
ep = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
api = HfApi(endpoint=ep)
rev = api.model_info(REPO).sha
files = [f for f in api.list_repo_tree(REPO, revision=rev, recursive=True) if getattr(f, "size", None) is not None]


def get(f):
    out = os.path.join(ck, f.path)
    if os.path.exists(out) and os.path.getsize(out) == f.size:
        return "skip"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    url = f"{ep}/{REPO}/resolve/{rev}/{urllib.request.quote(f.path)}"
    for i in range(6):
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=600) as r, open(out + ".part", "wb") as fo:
                for c in iter(lambda: r.read(1 << 24), b""):
                    h.update(c)
                    fo.write(c)
            if os.path.getsize(out + ".part") != f.size or (f.lfs is not None and h.hexdigest() != f.lfs.sha256):
                raise IOError("size/sha256 mismatch")
            os.replace(out + ".part", out)
            return "ok"
        except Exception as e:
            err = repr(e)
            time.sleep(10 * (i + 1))
    return "FAIL " + err


t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(get, files))
spm = os.path.join(dest, "paligemma_tokenizer.model")
if not os.path.exists(spm):
    urllib.request.urlretrieve(SPM_URL, spm + ".part")
    os.replace(spm + ".part", spm)
rec = {"repo": REPO, "revision": rev, "files": {f.path: f.size for f in files}, "results": dict(zip([f.path for f in files], res)),
       "tokenizer": SPM_URL, "tokenizer_sha256": hashlib.sha256(open(spm, "rb").read()).hexdigest(),
       "wall_s": round(time.time() - t0, 1)}
json.dump(rec, open(os.path.join(dest, "ASSETS.json"), "w"), indent=1)
bad = [p for p, r in rec["results"].items() if r.startswith("FAIL")]
print("###ASSETS###" + json.dumps({"revision": rev, "files": len(files), "bytes": sum(f.size for f in files), "failed": bad}))
sys.exit(1 if bad else 0)
