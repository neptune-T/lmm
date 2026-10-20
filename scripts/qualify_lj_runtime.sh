#!/usr/bin/env bash
# One bounded robot-loading qualification on physical GPU7. No data collection or training.
set -euo pipefail
ROOT=${LMM_REBUILD_ROOT:-/cluster/home/gaomingju/workspace/lmm/lmm}
CODE=$ROOT/code
R=$ROOT/runtime
RUN=$R/rebuild_20261020
export LMM_REPO=$CODE
export HABITAT_LAB=$R/home-robot/src/third_party/habitat-lab/habitat-lab
export ALPHABOT2_ASSETS=$R/alphabot2_assets
export HF_HOME=$ROOT/hf HF_SYNC_STAGE=$ROOT/backups/stage HF_TOKEN_PATH=$HOME/.config/lmm/hf_token
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
test -f "$RUN/habitat_env_backup_verified.json"
test -f "$RUN/robot_backup_verified.json"
# Physical index7 is passed to Habitat's EGL device selector. Torch in this environment is CPU-only.
unset CUDA_VISIBLE_DEVICES
export GPU=7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
FREE=$(nvidia-smi -i 7 --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$FREE" -lt 8192 ]; then
  echo "BLOCKED: GPU7 has less than 8GiB free for the bounded load check"; exit 1
fi
bash "$CODE/configs/alphabot2_habitat/install_overlay.sh" >"$RUN/overlay_install.log" 2>&1
"$ROOT/envs/backup/bin/python" - "$ROOT" <<'PY'
import hashlib, json, pathlib, sys
r = pathlib.Path(sys.argv[1])
p = r / "runtime/home-robot/src/third_party/habitat-lab/habitat-lab/habitat/config/benchmark/ovmm/ovmm_alphabot2.yaml"
p.write_text(p.read_text().replace("/workspace/runtime/alphabot2_assets", str(r / "runtime/alphabot2_assets")))
meta = {"change": "absolute asset paths only; original robot/task/physics parameters retained",
        "generated_config": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
(r / "runtime/rebuild_20261020/overlay_provenance.json").write_text(json.dumps(meta, indent=2))
PY
rc=0
ALPHABOT2_URDF="$ALPHABOT2_ASSETS/alphabot2/alphabot2.urdf" \
  "$ROOT/envs/home_robot39/bin/python" "$CODE/scripts/alphabot2_habitat/load_probe.py" \
  >"$RUN/robot_load_probe.log" 2>&1 || rc=$?
printf '{"exit_code":%d,"gpu_physical_index":7,"scope":"robot load only; not OVMM task smoke"}\n' \
  "$rc" >"$RUN/robot_load_probe_status.json"
"$ROOT/envs/backup/bin/python" "$CODE/scripts/hf_sync.py" --tag lj_20261020_robot_load \
  "$RUN/robot_load_probe.log" "$RUN/robot_load_probe_status.json" "$RUN/overlay_install.log" \
  "$RUN/overlay_provenance.json" "$HABITAT_LAB/habitat/config/benchmark/ovmm/ovmm_alphabot2.yaml" \
  "$ALPHABOT2_ASSETS/alphabot2.physics_config.json" --receipt "$RUN/robot_load_backup_verified.json"
exit "$rc"
