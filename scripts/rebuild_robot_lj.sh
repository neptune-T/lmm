#!/usr/bin/env bash
# Rebuild the existing robot asset contract from its original pinned upstream sources.
set -euo pipefail
ROOT=${LMM_REBUILD_ROOT:-/cluster/home/gaomingju/workspace/lmm/lmm}
CODE=$ROOT/code
R=$ROOT/runtime
RUN=$R/rebuild_20261020
mkdir -p "$RUN" "$R/deps" "$R/archive" "$ROOT/logs"
exec 9>"$RUN/robot_setup.lock"
flock -n 9 || exit 1
exec >>"$ROOT/logs/rebuild_robot.log" 2>&1
export GIT_AUTHOR_NAME="LMM Recovery" GIT_AUTHOR_EMAIL="lmm-recovery@localhost"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HOME=$ROOT/hf HF_SYNC_STAGE=$ROOT/backups/stage HF_TOKEN_PATH=$HOME/.config/lmm/hf_token
export PYTHONPATH=$CODE ALPHABOT2_DEPS=$R/deps
trap 'printf "%s BLOCKED robot_setup line=%s\n" "$(date -Is)" "$LINENO"' ERR
clone_sparse() {
  local repo=$1 name=$2 rev=$3
  shift 3
  if [ ! -d "$R/deps/$name/.git" ]; then
    git clone --filter=blob:none --no-checkout "https://github.com/fiveages-sim/$repo.git" "$R/deps/$name"
  fi
  git -C "$R/deps/$name" sparse-checkout set "$@"
  git -C "$R/deps/$name" checkout --detach "$rev"
  test "$(git -C "$R/deps/$name" rev-parse HEAD)" = "$rev"
}
clone_sparse robot_descriptions robot_descriptions bd491829961a8562306515001103af3b3737fbc7 \
  humanoid/Ai2Robotics/ai2_bot2_description humanoid/Realman/aidal_description manipulator/Realman/rm75_description
clone_sparse robot-descriptions-common rd_common 92fa1ebf7b7b6a36f8e277dda48706903e57101d \
  gripper/inspire_description sensor_models
E=$ROOT/envs/robot_tools
[ -x "$E/bin/python" ] || python3 -m venv "$E"
"$E/bin/python" -m pip install xacro==2.1.1 numpy==1.26.4
"$E/bin/python" -m lmm.embodiments.alphabot2.habitat.build_urdf "$R/alphabot2_assets/alphabot2"
"$E/bin/python" "$CODE/scripts/alphabot2_habitat/urdf_compare.py" \
  "$CODE/lmm/embodiments/alphabot2/habitat/alphabot2.urdf" "$R/alphabot2_assets/alphabot2/alphabot2.urdf" \
  >"$RUN/robot_urdf_comparison.jsonl"
"$E/bin/python" - "$R/alphabot2_assets" <<'PY'
import pathlib, sys
for p in pathlib.Path(sys.argv[1]).rglob("*"):
    if p.is_file():
        with p.open("rb") as f:
            if f.read(128).startswith(b"version https://git-lfs.github.com/spec"):
                raise SystemExit(f"LFS pointer not downloaded: {p}")
print("ROBOT_GEOMETRY_PRESENT")
PY
"$E/bin/python" -m pip freeze >"$RUN/robot_tools.pip-freeze.txt"
tar czf "$R/archive/alphabot2_assets.tar.gz.partial" -C "$R" alphabot2_assets
mv "$R/archive/alphabot2_assets.tar.gz.partial" "$R/archive/alphabot2_assets.tar.gz"
"$ROOT/envs/backup/bin/python" "$CODE/scripts/hf_sync.py" --tag lj_20261020_robot_assets \
  "$R/archive/alphabot2_assets.tar.gz" "$RUN/robot_urdf_comparison.jsonl" \
  "$RUN/robot_tools.pip-freeze.txt" --receipt "$RUN/robot_backup_verified.json"
printf '%s ROBOT_ASSETS_BACKUP_VERIFIED; live simulator checks still pending\n' "$(date -Is)"
