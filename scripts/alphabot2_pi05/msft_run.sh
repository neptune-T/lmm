#!/bin/bash
# M-SFT launcher (single GPU). Args: CONFIG GPU LOG
set -u
CFG=$1; G=$2; LOG=$3
Y=/mnt/cfs/h0ddz1/personal/liyanjun/yanjun; PY=$Y/envs/alphabot2_pi05/bin/python
export PI05_CKPT=/workspace/runtime/pi05/ckpt PI05_SPM=/workspace/runtime/pi05/paligemma_tokenizer.model TORCHDYNAMO_DISABLE=1 \
       PYTHONUNBUFFERED=1 HF_HOME=$Y/envs/_hf_home
used=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
[ "$G" != 3 ] && [ "$used" -lt 2000 ] || { echo "GPU $G not usable ($used MiB)" >> $LOG; exit 1; }
INIT=$($PY -c "import yaml; print(yaml.safe_load(open('$CFG'))['init_checkpoint'])")
echo "start $(date '+%F %T') commit $(git -C /workspace/lmm rev-parse --short HEAD) cfg $CFG gpu $G init $INIT init_sha $(sha256sum $INIT/model.safetensors | cut -c1-16)" >> $LOG
cd /workspace/lmm/scripts/alphabot2_pi05 && CUDA_VISIBLE_DEVICES=$G $PY sft_ovmm.py --config $CFG >> $LOG 2>&1
echo "exit $? $(date '+%F %T') init_sha_after $(sha256sum $INIT/model.safetensors | cut -c1-16)" >> $LOG
