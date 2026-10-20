#!/usr/bin/env bash
# Isolated CPU-only runtime installation. Does not run a simulator or collect episodes.
set -euo pipefail
ROOT=${LMM_REBUILD_ROOT:-/cluster/home/gaomingju/workspace/lmm/lmm}
CODE=$ROOT/code
R=$ROOT/runtime
RUN=$R/rebuild_20261020
mkdir -p "$RUN" "$R/archive" "$R/deps" "$ROOT/logs"
exec 9>"$RUN/habitat_setup.lock"
flock -n 9 || exit 1
exec >>"$ROOT/logs/rebuild_habitat.log" 2>&1
export GIT_AUTHOR_NAME="LMM Recovery" GIT_AUTHOR_EMAIL="lmm-recovery@localhost"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
export HF_HOME=$ROOT/hf HF_SYNC_STAGE=$ROOT/backups/stage HF_TOKEN_PATH=$HOME/.config/lmm/hf_token
event() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$RUN/habitat_events.log"; }
trap 'event "BLOCKED habitat_setup line=$LINENO exit=$?"' ERR
P=$ROOT/envs/backup/bin/python
event "START habitat source and environment restoration"
if [ ! -d "$R/home-robot/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/facebookresearch/home-robot.git "$R/home-robot"
fi
git -C "$R/home-robot" checkout ede6a67a2d0c0c8e12ad3b9726f330cea6cf90f5
git -C "$R/home-robot" submodule update --init src/third_party/habitat-lab
test "$(git -C "$R/home-robot/src/third_party/habitat-lab" rev-parse HEAD)" = b727ca9f7123101aaedb737ca9ccc1b153525dd9
event "PINNED_SOURCES_READY"
mkdir -p "$ROOT/tools"
if [ ! -x "$ROOT/tools/bin/micromamba" ]; then
  curl --fail --location --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest \
    -o "$R/archive/micromamba.tar.bz2"
  sha256sum "$R/archive/micromamba.tar.bz2" >"$R/archive/micromamba.tar.bz2.sha256"
  tar xjf "$R/archive/micromamba.tar.bz2" -C "$ROOT/tools" bin/micromamba
fi
export MAMBA_ROOT_PREFIX=$ROOT/envs/mamba
MM=$ROOT/tools/bin/micromamba
E=$ROOT/envs/home_robot39
if [ ! -f "$E/conda-meta/history" ]; then
  "$MM" create -y -p "$E" --override-channels -c conda-forge -c aihabitat \
    python=3.9 'habitat-sim=0.2.5' withbullet headless pip 'numpy=1.23.5'
fi
"$E/bin/python" -m pip install 'torch==1.13.1+cpu' 'torchvision==0.14.1+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
"$E/bin/python" -m pip install 'numpy==1.23.5' 'scipy==1.10.1' 'gym==0.23.0' \
  'hydra-core==1.3.2' 'omegaconf==2.3.0' yacs 'numpy-quaternion<2024' attrs \
  'numba<0.58' 'pillow<10' imageio imageio-ffmpeg matplotlib opencv-python-headless==4.10.0.84 \
  tqdm pyyaml tensorboard==2.8.0 protobuf==3.20.1 'moviepy<2' webdataset==0.1.40 \
  lmdb einops 'gymnasium<1' 'wandb<0.17' xacro trimesh==3.23.5 pybullet \
  ifcfg 'faster_fifo>=1.4.2' 'threadpoolctl>=3.1.0'
HL=$R/home-robot/src/third_party/habitat-lab
"$E/bin/python" -m pip install --no-deps -e "$HL/habitat-lab" -e "$HL/habitat-baselines"
"$E/bin/python" -c 'import habitat_sim, habitat, habitat_baselines, torch; print("IMPORT_OK", habitat_sim.__version__, torch.__version__)'
"$MM" list -p "$E" --explicit >"$R/archive/home_robot39.conda-explicit.txt"
"$E/bin/python" -m pip freeze >"$R/archive/home_robot39.pip-freeze.txt"
event "HABITAT_IMPORT_OK"
tar czf "$R/archive/home_robot39.tar.gz.partial" -C "$ROOT/envs" home_robot39
mv "$R/archive/home_robot39.tar.gz.partial" "$R/archive/home_robot39.tar.gz"
sha256sum "$R/archive/home_robot39.tar.gz" >"$R/archive/home_robot39.tar.gz.sha256"
# Bundle sources separately from data, which has its own stage-by-stage verified backups.
tar czf "$R/archive/home_robot_sources.tar.gz" --exclude=.git --exclude=home-robot/data -C "$R" home-robot
"$P" "$CODE/scripts/hf_sync.py" --tag lj_20261020_habitat_env \
  "$R/archive/home_robot39.tar.gz" "$R/archive/home_robot39.tar.gz.sha256" \
  "$R/archive/home_robot39.conda-explicit.txt" "$R/archive/home_robot39.pip-freeze.txt" \
  "$R/archive/home_robot_sources.tar.gz" --receipt "$RUN/habitat_env_backup_verified.json"
event "HABITAT_ENV_BACKUP_VERIFIED; robot overlay and live smoke still pending"
