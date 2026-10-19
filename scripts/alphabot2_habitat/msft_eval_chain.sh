#!/bin/bash
# After M-SFT v1 finishes: serve its `last` on GPUS (default 0,1,2; never 3) and run, paired with v2-B, (a) PA-OG +
# base lock on train20 (seeds 0, 1) / heldout20 / val30 (seed 0) and (b) the 240-job manipulation-from-arrival diagnostic.
# Clients are spread round-robin over the servers. Live logs in /dev/shm. ~33 sim workers (memory budget ~180 GB).
set -u
R=/workspace/runtime/runs/v2; Y=/mnt/cfs/h0ddz1/personal/liyanjun/yanjun; PY=$Y/envs/alphabot2_pi05/bin/python
OUTSFT=$Y/lmm_runtime/sft/msft_v1; CK=$OUTSFT/last; GPUS=${GPUS:-0,1,2}; IFS=, read -ra G <<< "$GPUS"; NG=${#G[@]}
S=/workspace/lmm/scripts/alphabot2_habitat; P5=/workspace/lmm/scripts/alphabot2_pi05
L=$R/msft_v1/chain.log
until grep -q "^exit" $R/msft_v1/train.log; do sleep 120; done
grep -q "^exit 0" $R/msft_v1/train.log || { echo "M-SFT did not exit 0; chain stops $(date '+%F %T')" >> $L; exit 1; }
echo "M-SFT done $(date '+%F %T'); last sha $(sha256sum $CK/model.safetensors | cut -c1-16)" >> $L
export PI05_CKPT=/workspace/runtime/pi05/ckpt PI05_SPM=/workspace/runtime/pi05/paligemma_tokenizer.model TORCHDYNAMO_DISABLE=1 \
       LMM_REPO=/workspace/lmm PYTHONUNBUFFERED=1 HF_HOME=$Y/envs/_hf_home
for g in "${G[@]}"; do
  [ "$g" != 3 ] || exit 1
  for k in $(seq 60); do used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " "); [ "$used" -lt 2000 ] && break; sleep 10; done
  (cd $P5 && CUDA_VISIBLE_DEVICES=$g setsid $PY policy_server.py --ckpt $CK --port $((5680 + g)) --dtype float32 > /dev/shm/msft_server_g$g.log 2>&1 &)
done
for g in "${G[@]}"; do
  for k in $(seq 120); do grep -q READY /dev/shm/msft_server_g$g.log 2>/dev/null && break; sleep 10; done
  grep -q READY /dev/shm/msft_server_g$g.log || { echo "server g$g not ready" >> $L; exit 1; }
done
echo "servers READY $(date '+%F %T') gpus $GPUS ckpt $CK" >> $L
HB='{"camera": "camera_top", "offsets": [5, 10, 15], "state": false}'
pa() {  # OUT UNITS SPLIT NW TAG SEED
  mkdir -p $1/traces; EP=$(cat $2 | tr '\n' ',')
  echo "msft lock $5 s$6 start $(date '+%F %T') commit $(git -C /workspace/lmm rev-parse --short HEAD) ckpt $CK" >> $1/README.txt
  for i in $(seq 0 $(($4 - 1))); do g=${G[$((i % NG))]}
    setsid env EPISODES="$EP" SPLIT=$3 OUT=$1 PORT=$((5680 + g)) GPU=$g WORKER=$i NWORKERS=$4 SEED_OFFSET=$6 BASE_LOCK=1 EXEC_STEPS=5 \
      HISTORY="$HB" POLICY_CKPT=$CK bash $S/run_habitat.sh $S/pa_eval.py > /dev/shm/msft_pa_$5_s$6_w$i.log 2>&1 < /dev/null &
  done; }
pa $R/pa_msft $R/units_train20.txt train 6 train20 0
pa $R/pa_msft $R/units_train20.txt train 6 train20 1
pa $R/pa_msft_heldout $R/units_heldout20.txt train 6 heldout20 0
pa $R/pa_msft_val $R/units_val30.txt val 6 val30 0
O=$R/manip_arrival_msft; mkdir -p $O/traces
echo "msft diag start $(date '+%F %T') commit $(git -C /workspace/lmm rev-parse --short HEAD) ckpt $CK jobs $R/manip_arrival/jobs.json" >> $O/README.txt
for i in $(seq 0 8); do g=${G[$((i % NG))]}
  setsid env JOBS=$R/manip_arrival/jobs.json OUT=$O PORT=$((5680 + g)) GPU=$g WORKER=$i NWORKERS=9 EXEC_STEPS=5 HISTORY="$HB" POLICY_CKPT=$CK \
    bash $S/run_habitat.sh $S/manip_from_arrival.py > /dev/shm/msft_diag_w$i.log 2>&1 < /dev/null &
done
echo "launched $(date '+%F %T')" >> $L
wait
echo "CHAIN_CLIENTS_DONE $(date '+%F %T') (servers left running)" >> $L
