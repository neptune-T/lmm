#!/bin/bash
# Manipulation frontier feasibility pilot (frozen v2-B + PA-OG/base lock; no training). Stages, each resumable:
#   servers  NS policy servers per GPU with --max-batch 1 (deterministic serving), GPUs ${GPUS:-0,1,2} (never 3)
#   source   PA episodes on the v2 train eval units (units_train20) x SOURCE_SEEDS, full event log + query snapshots
#   plan     frontier_plan.py -> roots + job files
#   verify   noise -1 per root (must reproduce its source bitwise); GATE: only passing roots are swept; < MIN_ROOTS stops
#   sweep    noises 0..7 per passing root;  dup: noise 0 for 6 roots again (determinism), then the analysis
# Live logs in /dev/shm; rows / traces / plan / analysis under $R/frontier_pilot.
set -u
R=/workspace/runtime/runs/v2; O=$R/frontier_pilot; Y=/mnt/cfs/h0ddz1/personal/liyanjun/yanjun; PY=$Y/envs/alphabot2_pi05/bin/python
S=/workspace/lmm/scripts/alphabot2_habitat; P5=/workspace/lmm/scripts/alphabot2_pi05
CK=/dev/shm/lmm_runtime/sft/v2B/best; GPUS=${GPUS:-0,1,2}; NS=${NS:-2}; CPS=${CPS:-2}; MIN_ROOTS=${MIN_ROOTS:-24}
SOURCE_SEEDS=${SOURCE_SEEDS:-100,101}
IFS=, read -ra G <<< "$GPUS"; mkdir -p $O; L=$O/chain.log
log() { echo "$(date '+%F %T') $*" >> $L; }
export PI05_CKPT=/workspace/runtime/pi05/ckpt PI05_SPM=/workspace/runtime/pi05/paligemma_tokenizer.model TORCHDYNAMO_DISABLE=1 \
       LMM_REPO=/workspace/lmm PYTHONUNBUFFERED=1 HF_HOME=$Y/envs/_hf_home
HB='{"camera": "camera_top", "offsets": [5, 10, 15], "state": false}'
# a container restart wipes /dev/shm: restore v2-B best from the verified step-10000 mirror (sha256 must match)
WANT=405f2d5d785b52af4d1f72b01500afa94cbf60c618f90ed3bbe529f882117fd7
if [ ! -f $CK/model.safetensors ]; then
  mkdir -p $(dirname $CK); cp -r /workspace/runtime/sft_mirror/v2B/ckpt_step0010000/policy $CK
  log "v2-B best restored from the mirror"
fi
[ "$(sha256sum $CK/model.safetensors | cut -c1-64)" = "$WANT" ] || { log "v2-B sha256 mismatch; refusing to run"; exit 1; }
log "start commit $(git -C /workspace/lmm rev-parse --short HEAD) ckpt $CK sha $(sha256sum $CK/model.safetensors | cut -c1-16) gpus $GPUS servers/gpu $NS clients/server $CPS"
PORTS=()
for g in "${G[@]}"; do
  [ "$g" != 3 ] || { log "GPU 3 refused"; exit 1; }
  for s in $(seq 0 $((NS - 1))); do
    p=$((5700 + 10 * g + s)); PORTS+=($p)
    pgrep -f "policy_server.py --ckpt $CK --port $p " >/dev/null && continue
    (cd $P5 && CUDA_VISIBLE_DEVICES=$g setsid $PY policy_server.py --ckpt $CK --port $p --dtype float32 --max-batch 1 > /dev/shm/fp_server_$p.log 2>&1 < /dev/null &)
    sleep 20
  done
done
for p in "${PORTS[@]}"; do for k in $(seq 120); do grep -q READY /dev/shm/fp_server_$p.log 2>/dev/null && break; sleep 10; done
  grep -q READY /dev/shm/fp_server_$p.log || { log "server $p not ready"; exit 1; }; done
log "servers READY ports ${PORTS[*]} (--max-batch 1)"
NC=$((${#PORTS[@]} * CPS))
clients() {  # MODE OUTDIR extra-env... : NC workers round-robin over the servers, wait
  local mode=$1 out=$2; shift 2; mkdir -p $out
  for i in $(seq 0 $((NC - 1))); do p=${PORTS[$((i % ${#PORTS[@]}))]}; g=$(( (p - 5700) / 10 ))
    setsid env MODE=$mode OUT=$out PORT=$p GPU=$g WORKER=$i NWORKERS=$NC HISTORY="$HB" POLICY_CKPT=$CK "$@" \
      bash $S/run_habitat.sh $S/frontier_pa.py > /dev/shm/fp_$(basename $out)_w$i.log 2>&1 < /dev/null &
  done; wait; }
# ---- source
IFS=, read -ra SS <<< "$SOURCE_SEEDS"
for sd in "${SS[@]}"; do
  log "source seed $sd start"
  clients source $O/source SEED=$sd UNITS="$(cat $R/units_train20.txt | tr '\n' ',')"
  log "source seed $sd done rows $(cat $O/source/source_w*.jsonl 2>/dev/null | wc -l)"
done
# ---- plan
$PY $S/frontier_plan.py --source-dir $O/source --out $O/plan >> $L 2>&1 || { log "plan failed"; exit 1; }
log "plan written"
# ---- verify (restore check) + gate
clients rollout $O/verify JOBS=$O/plan_verify.json
$PY $S/frontier_analysis.py --source $O/source --verify $O/verify --plan $O/plan --gpus ${#G[@]} --out $O/analysis_verify.json > /dev/shm/fp_av.log 2>&1
NPASS=$(python3 -c "import json; print(json.load(open('$O/analysis_verify.json'))['restore_check']['pass'])")
log "restore check pass $NPASS / $(python3 -c "import json; print(json.load(open('$O/analysis_verify.json'))['restore_check']['n'])")"
[ "$NPASS" -ge "$MIN_ROOTS" ] || { log "GATE: fewer than $MIN_ROOTS roots restore exactly; sweep NOT started"; exit 1; }
python3 - <<PYEOF
import json
ok = {v["root_id"] for v in json.load(open("$O/analysis_verify.json"))["restore_check"]["rows"] if v["bitwise_equal"]}
for name in ("sweep", "dup"):
    d = json.load(open(f"$O/plan_{name}.json")); d["jobs"] = [j for j in d["jobs"] if j["root_id"] in ok]
    json.dump(d, open(f"$O/plan_{name}_gated.json", "w"), indent=1)
PYEOF
log "sweep start (restore-verified roots only)"
clients rollout $O/sweep JOBS=$O/plan_sweep_gated.json
log "sweep done rows $(cat $O/sweep/rollout_w*.jsonl | wc -l)"
clients rollout $O/dup JOBS=$O/plan_dup_gated.json
log "dup done"
$PY $S/frontier_analysis.py --source $O/source --verify $O/verify --dup $O/dup --sweep $O/sweep --plan $O/plan --gpus ${#G[@]} --out $O/analysis.json > $O/analysis.log 2>&1
log "FRONTIER_PILOT_DONE (servers left running)"
