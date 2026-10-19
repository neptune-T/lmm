#!/bin/bash
# manip_from_arrival.py runner (policy servers + Habitat workers). Args: JOBS OUTDIR NW GPU[,GPU...] [CKPT]
# Never uses GPU 3 (keep-alive) or a GPU with > 2 GB in use.
set -u
JOBS=$1; O=$2; NW=$3; GPUS=$4; CK=${5:-/dev/shm/lmm_runtime/sft/v2B/best}
Y=/mnt/cfs/h0ddz1/personal/liyanjun/yanjun; PY=$Y/envs/alphabot2_pi05/bin/python
S=/workspace/lmm/scripts/alphabot2_habitat; P5=/workspace/lmm/scripts/alphabot2_pi05
export PI05_CKPT=/workspace/runtime/pi05/ckpt PI05_SPM=/workspace/runtime/pi05/paligemma_tokenizer.model TORCHDYNAMO_DISABLE=1 \
       LMM_REPO=/workspace/lmm PYTHONUNBUFFERED=1 HF_HOME=$Y/envs/_hf_home
HB=$($PY -c "import json, yaml; print(json.dumps(yaml.safe_load(open('/workspace/runtime/runs/v2/sft_B.yaml'))['history']))")
mkdir -p $O/traces
CKSHA=$(cat $CK/*.safetensors 2>/dev/null | sha256sum | cut -c1-16)
echo "start $(date '+%F %T') commit $(git -C /workspace/lmm rev-parse --short HEAD) jobs $(sha256sum $JOBS | cut -c1-16) ckpt $CK sha $CKSHA gpus $GPUS nw $NW history $HB" >> $O/README.txt
IFS=, read -ra G <<< "$GPUS"; TOT=$((${#G[@]} * NW)); i=0
for g in "${G[@]}"; do
  used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
  [ "$g" != 3 ] && [ "$used" -lt 2000 ] || { echo "GPU $g not usable ($used MiB)" >> $O/README.txt; exit 1; }
  (cd $P5 && CUDA_VISIBLE_DEVICES=$g setsid $PY policy_server.py --ckpt $CK --port $((5670 + g)) --dtype float32 > $O/server_g$g.log 2>&1 &)
done
for g in "${G[@]}"; do for k in $(seq 120); do grep -q READY $O/server_g$g.log 2>/dev/null && break; sleep 10; done; done
for g in "${G[@]}"; do
  for w in $(seq 0 $((NW - 1))); do
    env JOBS=$JOBS OUT=$O PORT=$((5670 + g)) GPU=$g WORKER=$i NWORKERS=$TOT EXEC_STEPS=5 HISTORY="$HB" POLICY_CKPT=$CK \
        bash $S/run_habitat.sh $S/manip_from_arrival.py > $O/w$i.log 2>&1 &
    i=$((i + 1))
  done
done
wait
for g in "${G[@]}"; do pkill -f "policy_server.py --ckpt $CK --port $((5670 + g))"; done
echo "MANIP_DONE $(date '+%F %T')" >> $O/README.txt
