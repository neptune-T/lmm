#!/bin/bash
# v2 A/B experiment driver (tmux; resumable). MODE=smoke runs the whole chain on a tiny manifest on one GPU;
# MODE=full: wait for >= MIN_ACCEPTED accepted trajectories -> frozen manifest v2 (H_exec calibration episodes
# excluded) -> sharded streaming conversion + merge (exact stats frozen to the manifest) -> interface gates
# (A and B configs) -> steering-weight sweep w in {1,2,4} (2k steps, sweep_decide.py) -> SFT A on GPU_A and B on GPU_B in parallel (same budget; resumable) -> per variant:
# train-set closed loop (20 trained-on units, traced) + the fixed OVMM-val eval set x 2 policy-noise seeds,
# H_exec frozen at 5 -> NAV metrics + funnel -> v2 comparison json. Collection keeps the other GPU.
#   MODE=smoke|full GPU_A=0 GPU_B=1 GPU_EVAL=2 bash v2_run.sh
set -u
MODE=${MODE:-full}; GPU_A=${GPU_A:-0}; GPU_B=${GPU_B:-1}; GPU_EVAL=${GPU_EVAL:-2}; MIN_ACCEPTED=${MIN_ACCEPTED:-2000}
R=/workspace/runtime/runs; S=/workspace/lmm/scripts/alphabot2_habitat; P5=/workspace/lmm/scripts/alphabot2_pi05
Y=/mnt/cfs/h0ddz1/personal/liyanjun/yanjun; PY=$Y/envs/alphabot2_pi05/bin/python; L=$Y/lmm_runtime/logs
TAG=$([ "$MODE" = smoke ] && echo v2smoke || echo v2)
DS=$Y/lmm_runtime/datasets; O=$R/$TAG; mkdir -p $O
export PI05_CKPT=/workspace/runtime/pi05/ckpt PI05_SPM=/workspace/runtime/pi05/paligemma_tokenizer.model TORCHDYNAMO_DISABLE=1 \
       LMM_REPO=/workspace/lmm PYTHONUNBUFFERED=1 HF_HOME=$Y/envs/_hf_home
ev() { echo "{\"time\": \"$(date '+%F %T')\", \"event\": \"$1\", \"mode\": \"$MODE\"${2:+, $2}}" | tee -a $O/events.jsonl; }
accepted() { python3 -c "import json; print(json.load(open('$R/collect_clean_v1/supervisor_status.json'))['accepted'])"; }

# ---------------------------------------------------------------- 1. frozen manifest
M=$R/manifests/$TAG.json
if [ ! -f $M ]; then
  if [ "$MODE" = full ]; then
    while [ "$(accepted)" -lt $MIN_ACCEPTED ]; do sleep 300; done
    CUT=$(date +%s)
  else
    CUT=$(date -d "2026-10-11 21:50:00" +%s)                        # small early snapshot
  fi
  python3 -c "import json; print('\n'.join(json.load(open('$R/eval_splits/val_r3_calib_eval.json'))['calibration']))" > $O/calibration_units.txt
  python3 $S/make_manifest.py --src $R/collect_clean_v1 --round $TAG --cutoff $CUT --max-train $([ "$MODE" = smoke ] && echo 8 || echo 0) \
     --out $M --supervisor-log $L/supervisor.log --extra-start "90,91@2026-10-11 20:43:50" --exclude-episodes $O/calibration_units.txt \
     | tee $O/manifest.log | cut -c1-400
  ev manifest_frozen "\"n_train\": $(python3 -c "import json; print(json.load(open('$M'))['n_train'])"), \"n_val\": $(python3 -c "import json; print(json.load(open('$M'))['n_val'])")"
fi
NT=$(python3 -c "import json; print(json.load(open('$M'))['n_train'])"); NV=$(python3 -c "import json; print(json.load(open('$M'))['n_val'])")

# ---------------------------------------------------------------- 2. sharded conversion + merge
convert() {  # split nshards
  local split=$1 n=$2 out=$DS/alphabot2_ovmm_${TAG}_$1
  [ -f $out/meta/exact_stats.json ] && [ -f $out/ovmm_episode_index.json ] && return 0
  for k in $(seq 0 $((n - 1))); do
    $PY $P5/ovmm_to_lerobot.py --src $R/collect_clean_v1 --manifest $M --split $split --shard $k/$n --streaming --encoder-threads 4 \
        --no-stats --out $DS/${TAG}_shards/${split}_$k > $L/${TAG}_convert_${split}_$k.log 2>&1 &
  done
  wait
  $PY $P5/merge_shards.py --shards $(for k in $(seq 0 $((n - 1))); do printf "$DS/${TAG}_shards/${split}_$k "; done) \
      --out $out --manifest $M --split $split > $L/${TAG}_merge_$split.log 2>&1 || { ev BLOCKED "\"gate\": \"CONVERT\", \"split\": \"$split\""; exit 1; }
  grep MERGE $L/${TAG}_merge_$split.log
}
NS=$([ "$MODE" = smoke ] && echo 2 || echo 24)
convert train $NS; convert val $([ $NV -lt 4 ] && echo 1 || echo 2)
ev converted

# ---------------------------------------------------------------- 3. configs + interface gates
for v in A B; do
  C=$O/sft_$v.yaml
  sed -e "s#alphabot2_ovmm_v2_train#alphabot2_ovmm_${TAG}_train#; s#alphabot2_ovmm_v2_val#alphabot2_ovmm_${TAG}_val#; s#/dev/shm/lmm_runtime/sft/v2$v#/dev/shm/lmm_runtime/sft/${TAG}$v#" \
      /workspace/lmm/configs/alphabot2_pi05/sft_v2$v.yaml > $C
  [ "$MODE" = smoke ] && sed -i "s/^steps: .*/steps: 20/; s/^val_every: .*/val_every: 10/; s/^ckpt_every: .*/ckpt_every: 10/" $C
  gate_ok() { python3 -c "import json,sys; g=json.load(open('$O/gate_$1/interface_gate.json')); sys.exit(0 if g['pass'] is True and all(g['checks'].values()) else 1)" 2>/dev/null; }
  if ! gate_ok $v; then
    (cd $P5 && CUDA_VISIBLE_DEVICES=$GPU_EVAL $PY interface_gate.py --config $C --collect $R/collect_clean_v1 --h-exec 5 --out $O/gate_$v > $L/${TAG}_gate_$v.log 2>&1)
    python3 -c "import json; g=json.load(open('$O/gate_$v/interface_gate.json')); print('gate $v', g['pass'], [k for k,x in g['checks'].items() if not x])"
    gate_ok $v || { ev BLOCKED "\"gate\": \"INTERFACE\", \"variant\": \"$v\""; exit 1; }   # top-level pass only
  fi
done
ev gates_pass

# ---------------------------------------------------------------- 3b. steering-weight sanity sweep (A config, short)
# w in {1, 2, 4}: omega flow weight w, steering auxiliary weight w/4; decision rule in sweep_decide.py
SW=$O/sweep; mkdir -p $SW
if [ ! -f $SW/decision.json ]; then
  for w in 1 2 4; do
    $PY - "$O/sft_A.yaml" "$SW/sft_w$w.yaml" "$w" "$TAG" "$MODE" <<'PY'
import sys, yaml
src, dst, w, tag, mode = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
c = yaml.safe_load(open(src))
c["loss"] = {"dim_weights": {"base.omega": w}, "steer_aux": {"weight": w / 4, "steps": 10, "dims": ["base.v", "base.omega"]}}
c["steps"] = 20 if mode == "smoke" else 2000
c["val_every"] = c["ckpt_every"] = 10 if mode == "smoke" else 500
c["out"] = f"/dev/shm/lmm_runtime/sft/{tag}sweep_w{w:g}"
c["name"] = f"sft_{tag}sweep_w{w:g}"
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
PY
  done
  sweep_one() {  # w gpu
    local out=/dev/shm/lmm_runtime/sft/${TAG}sweep_w$1
    for attempt in 1 2 3; do
      [ -f $out/DONE ] && return 0
      (cd $P5 && CUDA_VISIBLE_DEVICES=$2 $PY sft_ovmm.py --config $SW/sft_w$1.yaml >> $L/${TAG}_sweep_w$1.log 2>&1)
    done
  }
  if [ "$MODE" = full ]; then
    echo "{\"release_gpus\": [$GPU_A, $GPU_B, $GPU_EVAL]}" > $R/collect_clean_v1/collect_control.json; sleep 200
    sweep_one 1 $GPU_A & sweep_one 2 $GPU_B & sweep_one 4 $GPU_EVAL & wait
    echo "{\"release_gpus\": [$GPU_A, $GPU_B]}" > $R/collect_clean_v1/collect_control.json
  else
    for w in 1 2 4; do sweep_one $w $GPU_EVAL; done
  fi
  for w in 1 2 4; do
    E=$SW/w$w; mkdir -p $E; ck=/dev/shm/lmm_runtime/sft/${TAG}sweep_w$w/best; port=$((5630 + w))
    (cd $P5 && CUDA_VISIBLE_DEVICES=$GPU_EVAL setsid $PY policy_server.py --ckpt $ck --port $port > $E/server.log 2>&1 &)
    for i in $(seq 120); do grep -q READY $E/server.log 2>/dev/null && break; sleep 10; done
    (cd $P5 && CUDA_VISIBLE_DEVICES=$GPU_EVAL $PY diag_audit.py --ckpt $ck --train $DS/alphabot2_ovmm_${TAG}_train --val $DS/alphabot2_ovmm_${TAG}_val \
        --port $port --out $E/audit.json --n $([ "$MODE" = smoke ] && echo 20 || echo 300) --splits val > $E/audit.log 2>&1)
    U=$(cut -d, -f1-$([ "$MODE" = smoke ] && echo 1 || echo 10) $R/diag/sanity_v1/train_units.txt)
    for wk in 0 1 2 3 4; do
      EPISODES=$U SPLIT=train PORT=$port OUT=$E/cl GPU=$GPU_EVAL WORKER=$wk NWORKERS=5 EXEC_STEPS=5 SEED_OFFSET=0 \
      MAX_STEPS=$([ "$MODE" = smoke ] && echo 200 || echo 1500) TRACE_DIR=$E/traces bash $S/run_habitat.sh $S/eval_policy.py > $E/cl_w$wk.log 2>&1 &
    done
    wait
    pkill -f "policy_server.py --ckpt $ck --port $port"
    $PY $P5/diag_closed_loop.py --traces $E/traces --manifest $R/manifests/sanity_v1.json --out $E/closed_loop.json > /dev/null 2>&1
  done
  python3 $P5/sweep_decide.py --root $SW --weights 1 2 4 --out $SW/decision.json | tail -40
  [ -f $SW/decision.json ] || { ev BLOCKED "\"gate\": \"SWEEP\""; exit 1; }
  ev sweep_done "\"chosen_w\": $(python3 -c "import json; print(json.load(open('$SW/decision.json'))['chosen_w'])")"
fi
for v in A B; do                                                  # the chosen steering loss for both variants
  $PY - "$O/sft_$v.yaml" "$SW/decision.json" <<'PY'
import sys, json, yaml
c = yaml.safe_load(open(sys.argv[1]))
c["loss"] = json.load(open(sys.argv[2]))["loss"]
yaml.safe_dump(c, open(sys.argv[1], "w"), sort_keys=False)
PY
done

# ---------------------------------------------------------------- 4. SFT A and B (same budget), resumable
sft() {  # variant gpu
  local v=$1 g=$2 out=/dev/shm/lmm_runtime/sft/${TAG}$1
  for attempt in 1 2 3 4 5; do
    [ -f $out/DONE ] && return 0
    (cd $P5 && CUDA_VISIBLE_DEVICES=$g $PY sft_ovmm.py --config $O/sft_$v.yaml >> $L/${TAG}_sft_$v.log 2>&1)
    rc=$?; ev sft_exit "\"variant\": \"$v\", \"rc\": $rc, \"attempt\": $attempt"
    [ $rc -eq 3 ] && return 3
  done
  [ -f $out/DONE ]
}
if [ "$MODE" = full ]; then
  echo "{\"release_gpus\": [$GPU_A, $GPU_B]}" > $R/collect_clean_v1/collect_control.json   # collection keeps GPU_EVAL
  sleep 200
  sft A $GPU_A & PA=$!; sft B $GPU_B & PB=$!
  wait $PA; RA=$?; wait $PB; RB=$?
  echo '{"release_gpus": []}' > $R/collect_clean_v1/collect_control.json
else
  sft A $GPU_EVAL; RA=$?; sft B $GPU_EVAL; RB=$?
fi
ev sft_done "\"rc_A\": $RA, \"rc_B\": $RB"

# ---------------------------------------------------------------- 5. eval (train-set closed loop + fixed OVMM-val set x 2 seeds)
TRAIN_UNITS=$(cat $R/diag/sanity_v1/train_units.txt)
VAL_UNITS=$(python3 -c "import json; print(','.join(json.load(open('$R/eval_splits/val_r3_calib_eval.json'))['eval']))")
if [ "$MODE" = smoke ]; then TRAIN_UNITS=$(echo $TRAIN_UNITS | cut -d, -f1); VAL_UNITS=""; fi
evalv() {  # variant port
  local v=$1 port=$2 E=$O/eval_$1 ck=/dev/shm/lmm_runtime/sft/${TAG}$1/best hist=""
  [ "$v" = B ] && hist='{"camera": "camera_top", "offsets": [5, 10, 15], "state": true}'
  mkdir -p $E
  (cd $P5 && CUDA_VISIBLE_DEVICES=$GPU_EVAL setsid $PY policy_server.py --ckpt $ck --port $port > $E/server.log 2>&1 &)
  for i in $(seq 120); do grep -q READY $E/server.log 2>/dev/null && break; sleep 10; done
  local NW=5 MAXS=$([ "$MODE" = smoke ] && echo 300 || echo 3000)
  run() {  # name units split seed trace
    for w in $(seq 0 $((NW - 1))); do
      EPISODES=$2 SPLIT=$3 PORT=$port OUT=$E/$1 GPU=$GPU_EVAL WORKER=$w NWORKERS=$NW EXEC_STEPS=5 SEED_OFFSET=$4 MAX_STEPS=$MAXS \
      HISTORY="$hist" ${5:+TRACE_DIR=$E/traces_$1 VIDEO_EVERY=3} bash $S/run_habitat.sh $S/eval_policy.py > $E/$1_w$w.log 2>&1 &
    done
    wait
    python3 $S/eval_summary.py $E/$1 $2 > /dev/null
  }
  run train_cl $TRAIN_UNITS train 0 trace
  if [ -n "$VAL_UNITS" ]; then run val_pass0 $VAL_UNITS val 0 trace; run val_pass1 $VAL_UNITS val 1; fi
  pkill -f "policy_server.py --ckpt $ck --port $port"
  $PY $P5/diag_closed_loop.py --traces $E/traces_train_cl --manifest $R/manifests/sanity_v1.json --out $E/closed_loop_train.json > /dev/null 2>&1
  [ -d $E/traces_val_pass0 ] && $PY $P5/diag_closed_loop.py --traces $E/traces_val_pass0 --manifest $R/manifests/sanity_v1.json --out $E/closed_loop_val.json > /dev/null 2>&1
}
[ -f /dev/shm/lmm_runtime/sft/${TAG}A/DONE ] && evalv A 5620
[ -f /dev/shm/lmm_runtime/sft/${TAG}B/DONE ] && evalv B 5621
$PY $P5/v2_compare.py --root $O --baseline $R/diag/sanity_v1 --out $O/v2_comparison.json > $O/v2_comparison.txt 2>&1
cat $O/v2_comparison.txt
ev V2_DONE
