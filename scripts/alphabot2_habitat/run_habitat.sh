#!/usr/bin/env bash
# Run a script in the HomeRobot Habitat env with the patched source trees FIRST on PYTHONPATH
# (the conda env's site-packages carries an older, unpatched habitat) and the lmm repo importable.
set -euo pipefail
ROOT=${OVMM_SERVER_ROOT:-/nix/yanjun/zts/lmm}
HR=${OVMM_HOME_ROBOT_ROOT:-$ROOT/home-robot}
HL="$HR/src/third_party/habitat-lab"
export LMM_REPO=${LMM_REPO:-/nix/yanjun/zts/alphabot2/lmm_repo}
export PYTHONPATH="$HL/habitat-lab:$HL/habitat-baselines:$HR/src:$LMM_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$HR"          # OVMM configs use data/ paths relative to home-robot
exec "$ROOT/envs/home_robot39/bin/python" "$@"
