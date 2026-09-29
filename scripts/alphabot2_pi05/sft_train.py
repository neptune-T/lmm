"""lerobot-train entrypoint for AlphaBot pi0.5 SFT — PREPARED, NOT LAUNCHED in this phase.

Only difference from ``lerobot-train``: the checkpoint's saved preprocessor names
the gated ``google/paligemma-3b-pt-224`` tokenizer; here the tokenizer step is
given openpi's public PaliGemma SentencePiece model instead, encoded exactly as
at inference (``pi05_runtime.SentencePieceTokenizer``: BOS + pieces).

    python sft_train.py --config_path=configs/alphabot2_pi05_sft_pilot.json
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pi05_runtime import SPM, SentencePieceTokenizer  # noqa: E402

from lerobot.processor import tokenizer_processor as _tp  # noqa: E402

_orig_post_init = _tp.TokenizerProcessorStep.__post_init__


def _post_init(self):
    if self.tokenizer is None and self.tokenizer_name == "google/paligemma-3b-pt-224":
        self.tokenizer = SentencePieceTokenizer(SPM)
    _orig_post_init(self)


_tp.TokenizerProcessorStep.__post_init__ = _post_init

if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import main
    main()
