#!/bin/bash
# usage: launch_collect.sh <configs.json> <out_dir> <nworkers> <gpu_list e.g. "0,2"> [extra env...]
# One Isaac process per worker (CPU PhysX does not scale within a scene); workers are
# spread round-robin over the GPUs; BLAS/OpenMP threads pinned to 1 per process.
CFG=$1; OUT=$2; NW=$3; IFS=, read -ra GPUS <<< "$4"; shift 4
ROOT=/nix/yanjun/zts/alphabot2
mkdir -p $OUT/logs
for ((k=0; k<NW; k++)); do
  G=${GPUS[$((k % ${#GPUS[@]}))]}
  $ROOT/stage1/run_isaac.sh $G $OUT/logs/worker$(printf %02d $k).log $ROOT/lmm_repo/scripts/alphabot2_isaac/stage1/collect.py \
    CONFIGS=$CFG OUT=$OUT WORKER=$k NWORKERS=$NW OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$@"
  sleep 3
done
