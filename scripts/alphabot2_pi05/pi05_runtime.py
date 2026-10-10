"""Load the public AlphaBot pi0.5 checkpoint exactly as trained (lerobot 0.6.1).

Only deviation from the checkpoint's own processor config: the tokenizer. The
config names the gated `google/paligemma-3b-pt-224`; we use openpi's public
PaliGemma SentencePiece model (gs://big_vision/paligemma_tokenizer.model) and
encode with BOS exactly like openpi's PaligemmaTokenizer.
"""
import time
import numpy as np
import sentencepiece as spm
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

import os

CKPT = os.environ.get("PI05_CKPT", "/workspace/runtime/pi05/ckpt")                 # public AlphaBot pi0.5
SPM = os.environ.get("PI05_SPM", "/workspace/runtime/pi05/paligemma_tokenizer.model")
CAMS = ("camera_front", "camera_left", "camera_right", "camera_top")


class SentencePieceTokenizer:
    def __init__(self, path):
        self.sp = spm.SentencePieceProcessor(model_file=path)
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else 0

    def __call__(self, text, max_length=200, truncation=True, padding="max_length", padding_side="right",
                 return_tensors="pt"):
        text = [text] if isinstance(text, str) else list(text)
        ids = [[self.sp.bos_id()] + self.sp.encode(t) for t in text]
        if truncation:
            ids = [x[:max_length] for x in ids]
        L = max_length if padding == "max_length" else max(len(x) for x in ids)
        inp = torch.full((len(ids), L), self.pad_id, dtype=torch.long)
        att = torch.zeros((len(ids), L), dtype=torch.long)
        for i, x in enumerate(ids):
            sl = slice(0, len(x)) if padding_side == "right" else slice(L - len(x), L)
            inp[i, sl] = torch.tensor(x)
            att[i, sl] = 1
        return {"input_ids": inp, "attention_mask": att}


class Pi05Runtime:
    def __init__(self, device="cuda"):
        cfg = PreTrainedConfig.from_pretrained(CKPT)
        cfg.compile_model = False          # inference only; avoid max-autotune compile
        cfg.gradient_checkpointing = False
        cfg.device = device
        self.cfg = cfg
        self.policy = PI05Policy.from_pretrained(CKPT, config=cfg)
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=CKPT,
            preprocessor_overrides={"tokenizer_processor": {"tokenizer": SentencePieceTokenizer(SPM)},
                                    "device_processor": {"device": device}})
        self.device = device

    @torch.no_grad()
    def infer(self, images: dict, state16, task: str, seed: int = 0):
        """images: {camera: HxWx3 uint8}; state16 in the dataset convention. Returns (50,16) actions."""
        batch = {"observation.state": torch.as_tensor(np.asarray(state16, np.float32))[None], "task": [task]}
        for c in CAMS:
            im = torch.as_tensor(np.ascontiguousarray(images[c]))
            batch[f"observation.images.{c}"] = im.permute(2, 0, 1)[None].float() / 255.0
        torch.manual_seed(seed)
        t = time.perf_counter()
        b = self.pre(batch)
        a = self.policy.predict_action_chunk(b)
        a = self.post(a)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        return a[0].float().cpu().numpy(), dt
