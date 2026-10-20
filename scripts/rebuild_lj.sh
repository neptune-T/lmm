#!/usr/bin/env bash
# CPU-only recovery preparation; execute inside server-side tmux. Never launches training.
set -euo pipefail
ROOT=${LMM_REBUILD_ROOT:-/cluster/home/gaomingju/workspace/lmm/lmm}
CODE=$ROOT/code
RUN=$ROOT/runtime/rebuild_20261020
mkdir -p "$RUN" "$ROOT/logs" "$ROOT/backups" "$ROOT/hf" "$ROOT/envs"
exec 9>"$RUN/preparation.lock"
flock -n 9 || { echo "Preparation already running"; exit 1; }
export HF_HOME=$ROOT/hf
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.config/lmm/hf_token}
export HF_SYNC_STAGE=$ROOT/backups/stage
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
exec >>"$ROOT/logs/rebuild_prepare.log" 2>&1
event() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$RUN/events.log"; }
trap 'event "BLOCKED preparation line=$LINENO exit=$?"' ERR
event "START CPU preparation; GPU7 reserved only as a requested device, not allocated"
if [ ! -x "$ROOT/envs/backup/bin/python" ]; then
  python3 -m venv "$ROOT/envs/backup"
fi
P=$ROOT/envs/backup/bin/python
"$P" -m pip install 'huggingface_hub==1.19.0' >"$ROOT/logs/backup_env_install.log" 2>&1
"$P" -m pip freeze >"$RUN/backup.pip-freeze.txt"
"$P" "$CODE/scripts/hf_sync.py" --tag lj_20261020_code --git "$CODE" \
  "$CODE/docs/STATUS.md" "$CODE/docs/recovery_20261020.md" \
  --receipt "$RUN/code_backup_verified.json"
event "CODE_BACKUP_VERIFIED"
if [ ! -f "$RUN/base_model_backup_verified.json" ]; then
  "$P" "$CODE/scripts/alphabot2_pi05/fetch_assets.py" "$ROOT/runtime/pi05"
  "$P" "$CODE/scripts/hf_sync.py" --tag lj_20261020_public_base \
    "$ROOT/runtime/pi05" --receipt "$RUN/base_model_backup_verified.json"
fi
event "BASE_MODEL_BACKUP_VERIFIED (public initial checkpoint, not recovered v2-B)"
if [ ! -f "$RUN/pi05_env_backup_verified.json" ]; then
  PI05_ENV="$ROOT/envs/alphabot2_pi05" PI05_CKPT="$ROOT/runtime/pi05/ckpt" \
    PI05_ARCHIVE_DIR="$ROOT/runtime/archive" PYTHON_BOOTSTRAP=python3 \
    bash "$CODE/scripts/alphabot2_pi05/build_pi05_env.sh"
  # No active writers to this environment. Archive permissions/symlinks; do not archive HF credentials.
  tar czf "$ROOT/runtime/archive/alphabot2_pi05.tar.gz.partial" -C "$ROOT/envs" alphabot2_pi05
  mv "$ROOT/runtime/archive/alphabot2_pi05.tar.gz.partial" "$ROOT/runtime/archive/alphabot2_pi05.tar.gz"
  sha256sum "$ROOT/runtime/archive/alphabot2_pi05.tar.gz" >"$ROOT/runtime/archive/alphabot2_pi05.tar.gz.sha256"
  "$P" "$CODE/scripts/hf_sync.py" --tag lj_20261020_pi05_env \
    "$ROOT/runtime/archive" --receipt "$RUN/pi05_env_backup_verified.json"
fi
event "PREPARATION_DONE; Habitat/assets/collection gates still required; no automatic SFT/RL"
