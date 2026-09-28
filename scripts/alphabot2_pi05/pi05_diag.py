"""Input-sensitivity diagnosis on training frames: which variant best reproduces recorded actions?"""
import json, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/nix/yanjun/zts/alphabot2/hf")
exec(open("/nix/yanjun/zts/alphabot2/hf/pi05_offline_check.py").read().split("rt = Pi05Runtime()")[0])
from pi05_runtime import Pi05Runtime, SentencePieceTokenizer, SPM
rt = Pi05Runtime()
tok = rt.pre.steps[[type(s).__name__ for s in rt.pre.steps].index("TokenizerProcessorStep")]
class NoBos(SentencePieceTokenizer):
    def __call__(self, text, **kw):
        out = super().__call__(text, **kw)
        ids, att = out["input_ids"], out["attention_mask"]
        ids = torch.cat([ids[:, 1:], torch.full((ids.shape[0], 1), self.pad_id)], 1)
        att = torch.cat([att[:, 1:], torch.zeros((att.shape[0], 1), dtype=att.dtype)], 1)
        return {"input_ids": ids, "attention_mask": att}
rng = np.random.default_rng(1)
lens = pd.Series(E).value_counts()
picks = []
for task in (0, 1, 3):
    eps = [e for e in lens.index if TI[E == e][0] == task and lens[e] >= 60]
    for e in rng.choice(eps, 3, replace=False):
        picks.append((int(e), int(rng.integers(5, lens[e] - 40))))
cache = {}
for e, fi in picks:
    cache[(e, fi)] = frame(e, fi)
def score(variant):
    errs0, errs, rot = [], [], []
    for e, fi in picks:
        idx = np.nonzero((E == e) & (F == fi))[0][0]
        ims = dict(cache[(e, fi)]); st = S[idx].copy(); task = tname[int(TI[idx])]
        if variant == "black_images": ims = {k: np.zeros_like(v) for k, v in ims.items()}
        if variant == "swap_wrists": ims["camera_left"], ims["camera_right"] = ims["camera_right"], ims["camera_left"]
        if variant == "wrong_task": task = "Close gripper and do nothing"
        if variant == "state_from_other_episode":
            st = S[np.nonzero(E == picks[(picks.index((e, fi)) + 1) % len(picks)][0])[0][0]]
        pred, _ = rt.infer(ims, st, task, seed=0)
        gt = A[idx:idx + 20]
        errs0.append(np.linalg.norm(pred[0, 7:10] - gt[0, 7:10]) * 1000)
        errs.append(np.mean(np.linalg.norm(pred[:20, 7:10] - gt[:, 7:10], axis=1)) * 1000)
    return {"first_step_right_mm": round(float(np.median(errs0)), 1), "chunk20_right_mm": round(float(np.median(errs)), 1)}
out = {}
for v in ("baseline", "black_images", "swap_wrists", "wrong_task", "state_from_other_episode"):
    out[v] = score(v); print(v, out[v], flush=True)
tok.input_tokenizer = NoBos(SPM)
out["no_bos"] = score("baseline"); print("no_bos", out["no_bos"], flush=True)
hold = []
for e, fi in picks:
    idx = np.nonzero((E == e) & (F == fi))[0][0]
    hold.append(np.mean(np.linalg.norm(A[idx:idx + 20, 7:10] - A[idx, 7:10], axis=1)) * 1000)
out["hold_baseline_chunk20_right_mm"] = round(float(np.median(hold)), 1)
print("###JSON###" + json.dumps(out))
