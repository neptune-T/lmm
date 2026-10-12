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


# --------------------------------------------------------------------------- padding contract of the SFT loss
# Chunk steps past the episode end (action_is_pad; lerobot clamps them to the last real frame) must give
# (1) no target supervision and (2) no information to the valid steps. lerobot 0.6.1's PI05 forward
# ignores action_is_pad, and its action tokens attend to each other bidirectionally, so padded noisy
# actions x_t = t*noise + (1-t)*a_last would reach valid steps through attention (and carry the episode's
# tail action). flow_losses is PI05Pytorch.forward (lerobot 0.6.1, mirrored line for line) with ONE change:
# the suffix pad mask is ~action_is_pad, so make_att_2d_masks removes padded keys for every query (the
# additive -2.38e38 makes their softmax weight exactly 0); padding is always a chunk tail, so the
# position ids of valid steps are unchanged. reduce_masked gives padded steps loss weight exactly 0.
# The interface gate verifies: valid outputs invariant to padded inputs, zero padded loss weight, and
# bit-equality with lerobot's forward when nothing is padded (the mirror is pinned to 0.6.1).
import lerobot  # noqa: E402
from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d  # noqa: E402
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

LEROBOT_MIRRORED = "0.6.1"
assert lerobot.__version__ == LEROBOT_MIRRORED, f"flow_losses mirrors lerobot {LEROBOT_MIRRORED}, found {lerobot.__version__}"


def suffix_forward(model, images, img_masks, tokens, masks, actions, noise, time, action_valid=None):
    """PI05Pytorch.forward (lerobot 0.6.1) returning (per-element losses, v_t); action_valid (B, chunk)
    bool masks padded action tokens out of attention. action_valid=None is lerobot's forward exactly."""
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, tokens, masks)
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(x_t, time)
    if action_valid is not None:
        suffix_pad_masks = suffix_pad_masks & action_valid.to(suffix_pad_masks.device).bool()
    if model.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
        suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    att_2d_masks_4d = prepare_attention_masks_4d(att_2d_masks)

    def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
        (_, suffix_out), _ = model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d, position_ids=position_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs], use_cache=False, adarms_cond=[None, adarms_cond])
        return suffix_out

    suffix_out = model._apply_checkpoint(forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond)
    suffix_out = suffix_out[:, -model.config.chunk_size:].to(dtype=torch.float32)
    v_t = model._apply_checkpoint(lambda o: model.action_out_proj(o), suffix_out)
    return F.mse_loss(u_t, v_t, reduction="none"), v_t


def flow_losses(policy, batch, action_is_pad=None, noise=None, time=None, return_v=False):
    """Per-(chunk step, action dim) flow-matching losses with padded action tokens masked out of attention.
    noise/time default to the policy's own samplers (as PI05Policy.forward draws them)."""
    images, img_masks = policy._preprocess_images(batch)
    tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action(batch)
    if noise is None:
        noise = policy.model.sample_noise(actions.shape, actions.device)
    if time is None:
        time = policy.model.sample_time(actions.shape[0], actions.device)
    valid = None if action_is_pad is None else ~action_is_pad.to(actions.device).bool()
    losses, v_t = suffix_forward(policy.model, images, img_masks, tokens, masks, actions, noise, time, valid)
    d = policy.config.output_features["action"].shape[0]
    return (losses[:, :, :d], v_t[:, :, :d]) if return_v else losses[:, :, :d]


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
    return reduce_masked(flow_losses(policy, batch, action_is_pad), action_is_pad)


class MaskedLoss(torch.nn.Module):
    """Wraps the policy so DDP sees one forward (gradient sync) while the loss masks padded targets."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch, action_is_pad):
        return masked_loss(self.policy, batch, action_is_pad)
