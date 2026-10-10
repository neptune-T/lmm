#!/bin/bash
# Install the AlphaBot 2 OVMM overlay ALONGSIDE Stretch and WXAI (nothing is replaced):
# the Hydra overlay next to benchmark/ovmm/ovmm.yaml and the physics config next to the URDF,
# plus habitat_dual_hand.patch on habitat-lab b727ca9 (idempotent): the grasp_mgr property follows
# active_grasp_manager_idx and grasp_iter yields both hands (OVMM's pick/hold measures follow the
# hand that grasps); left/right wrist RGB sensor configs (224x224).
set -euo pipefail
REPO=${LMM_REPO:-/workspace/lmm}
HAB=${HABITAT_LAB:-/workspace/runtime/home-robot/src/third_party/habitat-lab/habitat-lab}
ASSETS=${ALPHABOT2_ASSETS:-/workspace/runtime/alphabot2_assets}
HLROOT=$(dirname "$HAB")
P=$REPO/configs/alphabot2_habitat/habitat_dual_hand.patch
if (cd "$HLROOT" && git apply --reverse --check "$P" 2>/dev/null); then
  echo "habitat_dual_hand.patch already applied"
else
  (cd "$HLROOT" && git apply --check "$P" && git apply "$P") && echo "habitat_dual_hand.patch applied"
fi
cp $REPO/configs/alphabot2_habitat/ovmm_alphabot2.yaml $HAB/habitat/config/benchmark/ovmm/ovmm_alphabot2.yaml
cp $REPO/configs/alphabot2_habitat/alphabot2.physics_config.json $ASSETS/alphabot2.physics_config.json
sha256sum $HAB/habitat/config/benchmark/ovmm/ovmm_alphabot2.yaml $ASSETS/alphabot2.physics_config.json
