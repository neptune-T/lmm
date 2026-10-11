#!/bin/bash
# pi0.5 SFT / policy-server env on the user's CFS (NFS: symlinks + exec work; not bosfs, not /dev/shm).
# Python 3.12 venv; torch 2.8.0+cu126 (driver 535 OK, qualified on the old A800); lerobot[pi,dataset]==0.6.1.
# All caches/temp on CFS (root has ~1.5 GB). Verifies against the fetched public checkpoint.
#     PI05_ENV=/mnt/cfs/.../envs/alphabot2_pi05 bash build_pi05_env.sh
set -euxo pipefail
E=${PI05_ENV:-/mnt/cfs/h0ddz1/personal/liyanjun/yanjun/envs/alphabot2_pi05}
W=$(dirname "$E")/_build_alphabot2_pi05
mkdir -p "$W/tmp" "$W/pipcache"
export TMPDIR=$W/tmp PIP_CACHE_DIR=$W/pipcache HF_HOME=$W/hf_home
[ -x "$E/bin/python" ] || python3.12 -m venv "$E"
P=$E/bin/python
TUNA="-i https://pypi.tuna.tsinghua.edu.cn/simple"
$P -m pip install -q -U pip $TUNA
# cu126 wheel index: the NJU mirror of download.pytorch.org (same files; ~80 MB/s here vs throttled upstream)
TORCH_INDEX=${TORCH_INDEX:-https://mirrors.nju.edu.cn/pytorch/whl/cu126}
$P -m pip install -q "torch==2.8.0" "torchvision==0.23.0" --index-url $TORCH_INDEX \
  || $P -m pip install -q "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu126
$P -m pip install -q "lerobot[pi,dataset]==0.6.1" sentencepiece pyyaml av $TUNA
$P - <<PY
import torch, lerobot, transformers
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.configs.policies import PreTrainedConfig
cfg = PreTrainedConfig.from_pretrained("${PI05_CKPT:-/workspace/runtime/pi05/ckpt}")
print("ENV_OK torch", torch.__version__, "cuda_build", torch.version.cuda, "lerobot", lerobot.__version__,
      "transformers", transformers.__version__, "ckpt_type", cfg.type, "chunk", cfg.chunk_size,
      "images", list(cfg.image_features))
PY
$P -m pip freeze > "$(dirname "$E")/alphabot2_pi05.pip-freeze.txt"
mkdir -p /workspace/runtime/archive && cp "$(dirname "$E")/alphabot2_pi05.pip-freeze.txt" /workspace/runtime/archive/
echo "###PI05_ENV_DONE### $E"
