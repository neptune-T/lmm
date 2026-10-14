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
# Post-restore extras, once per restored env (host moves expose what the archive lacks):
#  * pip packages installed after the archive was made (archive/home_robot39.extra-requirements.txt;
#    tensorboard: habitat_baselines imports torch.utils.tensorboard at import time);
#  * the glvnd EGL dispatch libs (libEGL.so.1) when the host lacks them -- extracted from the archived
#    debs into the env's lib (the NVIDIA EGL vendor ICD comes from the driver mount).
if [ ! -f "$ENV/.extras_ok" ]; then
  (
    flock 9
    if [ ! -f "$ENV/.extras_ok" ]; then
      XR="$ROOT/archive/home_robot39.extra-requirements.txt"
      [ -f "$XR" ] && "$ENV/bin/python" -m pip install -q -r "$XR" -i https://pypi.tuna.tsinghua.edu.cn/simple >&2
      if ! ldconfig -p | grep -q "libEGL.so.1 " && [ ! -e "$ENV/lib/libEGL.so.1" ]; then
        T=$(mktemp -d /dev/shm/egl.XXXX)
        for d in "$ROOT"/archive/egl/*.deb; do dpkg -x "$d" "$T"; done
        cp -a "$T"/usr/lib/x86_64-linux-gnu/lib{EGL,GLdispatch}.so* "$ENV/lib/" && rm -rf "$T"
      fi
      touch "$ENV/.extras_ok"
    fi
  ) 9>"$(dirname "$ENV")/.restore.lock"
fi
# OVMM configs use data/ paths relative to the working directory: run from a fast data tree on
# /dev/shm (small files copied, HSSD object meshes symlinked to the bosfs archive), staged once.
RUN_DIR=${OVMM_RUN_DIR:-/dev/shm/hr}
if [ ! -f "$RUN_DIR/data/.staged" ]; then
  mkdir -p "$RUN_DIR"
  (
    flock 9
    [ -f "$RUN_DIR/data/.staged" ] || "$ENV/bin/python" "$LMM_REPO/scripts/alphabot2_habitat/stage_data.py" "$HR/data" "$RUN_DIR/data" >&2
  ) 9>"$RUN_DIR/.stage.lock"
fi
cd "$RUN_DIR"
exec "$ENV/bin/python" "$@"
