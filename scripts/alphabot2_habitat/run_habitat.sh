#!/usr/bin/env bash
# Run a script in the HomeRobot Habitat env with the patched source trees FIRST on PYTHONPATH
# (the conda env's site-packages carries an older, unpatched habitat) and the lmm repo importable.
set -euo pipefail
ROOT=${OVMM_SERVER_ROOT:-/workspace/runtime}
HR=${OVMM_HOME_ROBOT_ROOT:-$ROOT/home-robot}
HL="$HR/src/third_party/habitat-lab"
export LMM_REPO=${LMM_REPO:-/workspace/lmm}
export PYTHONPATH="$HL/habitat-lab:$HL/habitat-baselines:$HR/src:$LMM_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$HR"          # OVMM configs use data/ paths relative to home-robot
# The executable env runs from /dev/shm (bosfs has no symlinks and unreliable exec bits);
# /workspace keeps only its archive. Restore once per container, serialized across workers.
ENV=${HR_ENV:-/dev/shm/envs/home_robot39}
ARCH=${HR_ENV_ARCHIVE:-$ROOT/archive/home_robot39.tar.gz}
if [ ! -x "$ENV/bin/python" ]; then
  mkdir -p "$(dirname "$ENV")"
  (
    flock 9
    if [ ! -x "$ENV/bin/python" ]; then
      (cd "$(dirname "$ARCH")" && sha256sum -c "$(basename "$ARCH").sha256" >&2)
      tar xzf "$ARCH" -C "$(dirname "$ENV")"
    fi
  ) 9>"$(dirname "$ENV")/.restore.lock"
fi
exec "$ENV/bin/python" "$@"
