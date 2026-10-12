"""Shared construction of the AlphaBot 2 OVMM pi0.5 policy + processors (SFT, interface gate, server).

* image features in the PRETRAINED relative order (camera_left, camera_right, camera_top); camera_front
  is absent. lerobot 0.6.1 pi05 appends configured-but-missing cameras after the present ones with
  pad mask 0, and positions are cumsum(pad_mask) - 1, so "camera_front masked" and "camera_front absent"
  are numerically identical (the interface gate verifies this); absent is the fixed choice.
* state 16-D (checkpoint convention), action 18-D CartesianMobileAction18;
* normalization: the checkpoint's mapping (VISUAL identity, STATE/ACTION quantiles q01/q99) with the
  OVMM TRAIN dataset statistics for every dim (base v/omega have no checkpoint stats);
* tokenizer: openpi's public PaliGemma SentencePiece model.
"""
import os, sys

REPO = os.environ.get("LMM_REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lmm.embodiments.alphabot2 import ovmm_policy_io as IO  # noqa: E402
from pi05_runtime import SPM, SentencePieceTokenizer  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402

CAMERAS = tuple(c for c in IO.MODEL_CAMERA_ORDER if c in IO.CAMERA_MAP)          # camera_left, camera_right, camera_top


def build_config(init_ckpt, device="cuda", dtype="float32", gradient_checkpointing=True, h_exec=None):
    cfg = PreTrainedConfig.from_pretrained(init_ckpt)
    cfg.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
                          **{f"observation.images.{c}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, *IO.IMAGE_SHAPE[:2]))
                             for c in CAMERAS}}
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(len(IO.ACTION_LABELS),))}
    cfg.empty_cameras = 0
    cfg.compile_model = False
    cfg.gradient_checkpointing = gradient_checkpointing
    cfg.dtype = dtype
    cfg.device = device
    if h_exec is not None:
        cfg.n_action_steps = int(h_exec)              # recorded for transparency; eval executes H_exec explicitly
    return cfg


def build(cfg, weights, init_ckpt, stats, device="cuda"):
    """weights: dir to load parameters from (init checkpoint or a resume checkpoint)."""
    policy = PI05Policy.from_pretrained(weights, config=cfg).to(device)
    feats = {**cfg.input_features, **cfg.output_features}
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=init_ckpt, dataset_stats=stats,
        preprocessor_overrides={"device_processor": {"device": device},
                                "normalizer_processor": {"stats": stats, "features": feats, "norm_map": cfg.normalization_mapping},
                                "tokenizer_processor": {"tokenizer": SentencePieceTokenizer(SPM)}},
        postprocessor_overrides={"unnormalizer_processor": {"stats": stats, "features": cfg.output_features,
                                                            "norm_map": cfg.normalization_mapping}})
    return policy, pre, post


# --------------------------------------------------------------------------- padded-target-masked SFT loss
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402
import torch  # noqa: E402


def flow_losses(policy, batch):
    """Per-(chunk step, action dim) flow-matching losses, exactly as PI05Policy.forward computes them."""
    images, img_masks = policy._preprocess_images(batch)
    tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action(batch)
    noise = policy.model.sample_noise(actions.shape, actions.device)
    time = policy.model.sample_time(actions.shape[0], actions.device)
    losses = policy.model.forward(images, img_masks, tokens, masks, actions, noise, time)
    return losses[:, :, :policy.config.output_features["action"].shape[0]]


def reduce_masked(losses, action_is_pad=None):
    """Mean over valid (non-padded) chunk steps and all action dims; padded steps get weight exactly 0."""
    d = losses.shape[-1]
    if action_is_pad is None:
        valid = torch.ones(losses.shape[:2], device=losses.device)
    else:
        valid = (~action_is_pad.to(losses.device).bool()).float()
    w = valid[:, :, None]
    n = w.sum().clamp_min(1.0)
    loss = (losses * w).sum() / (n * d)
    per_dim = (losses * w).sum(dim=(0, 1)) / n
    return loss, {"loss_per_dim": per_dim.detach().float().cpu().numpy().tolist(), "loss": float(loss),
                  "valid_target_fraction": float(valid.mean())}


def masked_loss(policy, batch, action_is_pad=None):
    """PI05Policy.forward's flow-matching loss with chunk steps beyond the episode end EXCLUDED.

    lerobot 0.6.1 clamps chunk indices past the episode end to the last frame and flags them
    action_is_pad, but PI05Policy.forward ignores the flag, so padded copies of the final (post-release)
    action would be trained as real targets. Same preprocessing, noise/time sampling and model call as
    PI05Policy.forward; only the reduction differs. action_is_pad: (B, chunk) bool from the RAW batch."""
    return reduce_masked(flow_losses(policy, batch), action_is_pad)


class MaskedLoss(torch.nn.Module):
    """Wraps the policy so DDP sees one forward (gradient sync) while the loss masks padded targets."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch, action_is_pad):
        return masked_loss(self.policy, batch, action_is_pad)
