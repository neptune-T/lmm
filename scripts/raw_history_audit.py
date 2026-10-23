"""CPU training-contract checks; optionally audit every accepted raw episode in a closed shard.

This is not the converted-video/model interface gate. Those require the later frozen
dataset and remain mandatory before any training.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "alphabot2_pi05"))
import ovmm_model as M
from lmm.embodiments.alphabot2.ovmm_chain_check import lowering_chain
from lmm.embodiments.alphabot2.raw_qc import semantic_reject, quarantine

H = {"camera": "camera_top", "offsets": [5, 10, 15], "state": False}
cfg = M.build_config(os.environ["PI05_CKPT"], device="cpu", history=H)
assert cfg.input_features["observation.state"].shape == (16,)
assert cfg.output_features["action"].shape == (18,)
assert cfg.chunk_size == 50
assert len(cfg.image_features) == 6
ts = M.history_delta_timestamps({}, H)["observation.images.camera_top"]
assert ts == [-1.5, -1.0, -0.5, 0.0]
result = {"cpu_contract_pass": True, "history": H, "raw_episodes_checked": [],
          "not_yet_verified": ["decoded videos", "actual model forward", "optimizer resume", "training VRAM"]}
if len(sys.argv) > 1:
    root = Path(sys.argv[1])
    reject_semantic = "--quarantine-semantic" in sys.argv[2:]
    report = root / "raw_qc_report.json"
    rejected = (json.loads(report.read_text()).get("raw_episodes_rejected", {})
                if reject_semantic and report.exists() else {})
    rs = [json.loads(l) for p in root.glob("results*.jsonl") for l in p.read_text().splitlines()]
    for r in rs:
        if not r.get("accepted"):
            continue
        d = Path(r["dir"])
        with np.load(d / "lowdim.npz") as low, np.load(d / "images.npz") as ims, np.load(d / "metadata.npz") as meta:
            n = len(low["proprio"])
            assert n == r["steps"] and n > 15
            for k, shape in (("proprio", (n, 16)), ("action", (n, 18)), ("joint18", (n, 18))):
                assert low[k].shape == shape and np.isfinite(low[k]).all(), (d, k)
            assert np.allclose(np.diff(low["sim_time"]), 0.1, atol=1e-6, rtol=0)
            assert len(meta["stage"]) == n
            # Read cameras one at a time, bounding CPU RAM; all frames must be present.
            for cam in M.IO.CAMERA_MAP.values():
                im = ims[cam]
                assert im.shape == (n, 240, 320, 3) and im.dtype == np.uint8, (d, cam)
            head = ims["head_rgb"]
            for t in sorted(set([0, 1, 5, 10, 15, n // 2, n - 1])):
                indices = [max(0, t - off) for off in (15, 10, 5, 0)]
                batch = {"observation.images.camera_top": torch.from_numpy(head[indices])[None],
                         "observation.state": torch.from_numpy(M.IO.state_from_proprio(low["proprio"][t]))[None]}
                state = batch["observation.state"].clone()
                batch = M.apply_history(batch, H)
                assert torch.equal(batch["observation.state"], state)
                for off in H["offsets"]:
                    assert np.array_equal(batch[f"observation.images.camera_top_h{off}"][0].numpy(),
                                          head[max(0, t - off)])
            chain = lowering_chain(low, meta)
            if reject_semantic and semantic_reject(chain):
                rejected[r["episode_id"]] = {"reason": "cartesian_target_not_executed",
                                             "chain": chain}
                continue
            assert all(chain[k]["pass"] for k in ("lowering", "joint_tracking",
                       "fk_next_state_matches_label", "gripper", "base")), (d, chain)
            result["raw_episodes_checked"].append({"unit": r["episode_id"], "frames": n, "chain": chain})
    # Only publish exclusions after all other episodes and infrastructure checks pass.
    if reject_semantic:
        quarantine(root, rejected)
        result["raw_episodes_rejected"] = rejected
        report = root / "raw_qc_report.json"
        tmp = report.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(report)
print(json.dumps(result))
