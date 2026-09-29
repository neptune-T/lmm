"""SFT readiness dry run — NO training: one forward loss, no backward, no optimizer, no checkpoint.

Builds what lerobot-train would for configs/alphabot2_pi05_sft_pilot.json (dataset
with action chunks, pi0.5 from the public checkpoint with an 18-D action output,
processors with this dataset's normalization stats and the public tokenizer),
then evaluates the flow-matching loss on one small batch.
"""
import json, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pi05_runtime import CKPT, SPM, SentencePieceTokenizer  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402

ROOT = os.environ.get("PILOT_LEROBOT", "/nix/yanjun/zts/alphabot2/pilot_lerobot")
REPO_ID = "local/alphabot2_sim_pilot"
out = {}
cfg = PreTrainedConfig.from_pretrained(CKPT)
cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(18,))}
cfg.compile_model = False
cfg.device = "cuda"
meta = LeRobotDatasetMetadata(REPO_ID, root=ROOT)
dts = resolve_delta_timestamps(cfg, meta)
ds = LeRobotDataset(REPO_ID, root=ROOT, delta_timestamps=dts)
out["dataset"] = {"episodes": meta.total_episodes, "frames": meta.total_frames, "fps": meta.fps,
                  "action_stats_dim": len(meta.stats["action"]["mean"]),
                  "base_v_std": float(np.asarray(meta.stats["action"]["std"])[16]),
                  "base_omega_std": float(np.asarray(meta.stats["action"]["std"])[17])}
policy = PI05Policy.from_pretrained(CKPT, config=cfg).to("cuda")
feats = {**cfg.input_features, **cfg.output_features}
pre, post = make_pre_post_processors(
    cfg, pretrained_path=CKPT, dataset_stats=meta.stats,
    preprocessor_overrides={"device_processor": {"device": "cuda"},
                            "normalizer_processor": {"stats": meta.stats, "features": feats, "norm_map": cfg.normalization_mapping},
                            "tokenizer_processor": {"tokenizer": SentencePieceTokenizer(SPM)}},
    postprocessor_overrides={"unnormalizer_processor": {"stats": meta.stats, "features": cfg.output_features,
                                                        "norm_map": cfg.normalization_mapping}})
loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(0))
batch = next(iter(loader))
out["batch_shapes"] = {k: list(v.shape) for k, v in batch.items() if hasattr(v, "shape")}
policy.train()
t = time.perf_counter()
with torch.no_grad():
    loss, info = policy.forward(pre(batch))
out["forward_loss"] = float(loss)
out["forward_s"] = round(time.perf_counter() - t, 3)
out["loss_finite"] = bool(torch.isfinite(loss))
out["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
out["note"] = "forward only; no backward, no optimizer step, nothing saved"
print("###DRYRUN###" + json.dumps(out, default=str), flush=True)
