#!/bin/bash
# pa_eval.py runner. Args: UNITS_FILE SPLIT OUTDIR NW GPU[,GPU...] SEEDS(comma) [CKPT]
# One policy server per GPU; NW workers per GPU per seed pass. Never GPU 3 or a GPU with > 2 GB in use.
set -u
UF=$1; SPLIT=$2; O=$3; NW=$4; GPUS=$5; SEEDS=$6; CK=${7:-/dev/shm/lmm_runtime/sft/v2B/best}
Y=/mnt/cfs/h0ddz1/personal/liyanjun/yanjun; PY=$Y/envs/alphabot2_pi05/bin/python
S=/workspace/lmm/scripts/alphabot2_habitat; P5=/workspace/lmm/scripts/alphabot2_pi05
export PI05_CKPT=/workspace/runtime/pi05/ckpt PI05_SPM=/workspace/runtime/pi05/paligemma_tokenizer.model TORCHDYNAMO_DISABLE=1 \
       LMM_REPO=/workspace/lmm PYTHONUNBUFFERED=1 HF_HOME=$Y/envs/_hf_home
HB=$($PY -c "import json, yaml; print(json.dumps(yaml.safe_load(open('/workspace/runtime/runs/v2/sft_B.yaml'))['history']))")
mkdir -p $O/traces
CKSHA=$(cat $CK/*.safetensors 2>/dev/null | sha256sum | cut -c1-16)
echo "start $(date '+%F %T') commit $(git -C /workspace/lmm rev-parse --short HEAD) units $UF ($(sha256sum $UF | cut -c1-16)) split $SPLIT seeds $SEEDS ckpt $CK sha $CKSHA gpus $GPUS nw $NW history $HB" >> $O/README.txt
IFS=, read -ra G <<< "$GPUS"; IFS=, read -ra SD <<< "$SEEDS"
for g in "${G[@]}"; do
  used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
  [ "$g" != 3 ] && [ "$used" -lt 2000 ] || { echo "GPU $g not usable ($used MiB)" >> $O/README.txt; exit 1; }
  (cd $P5 && CUDA_VISIBLE_DEVICES=$g setsid $PY policy_server.py --ckpt $CK --port $((5680 + g)) --dtype float32 > $O/server_g$g.log 2>&1 &)
done
for g in "${G[@]}"; do for k in $(seq 120); do grep -q READY $O/server_g$g.log 2>/dev/null && break; sleep 10; done; done
EP=$(cat $UF | tr '\n' ',')
for sd in "${SD[@]}"; do
  TOT=$((${#G[@]} * NW)); i=0
  for g in "${G[@]}"; do
    for w in $(seq 0 $((NW - 1))); do
      env EPISODES="$EP" SPLIT=$SPLIT OUT=$O PORT=$((5680 + g)) GPU=$g WORKER=$i NWORKERS=$TOT SEED_OFFSET=$sd EXEC_STEPS=5 \
          HISTORY="$HB" POLICY_CKPT=$CK bash $S/run_habitat.sh $S/pa_eval.py > $O/w${i}_s$sd.log 2>&1 &
      i=$((i + 1))
    done
  done
done
wait
for g in "${G[@]}"; do pkill -f "policy_server.py --ckpt $CK --port $((5680 + g))"; done
echo "PA_DONE $(date '+%F %T')" >> $O/README.txt
