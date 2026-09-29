#!/bin/bash
# Install the AlphaBot 2 OVMM overlay ALONGSIDE Stretch and WXAI (nothing is replaced):
# the Hydra overlay next to benchmark/ovmm/ovmm.yaml and the physics config next to the URDF.
set -euo pipefail
REPO=${LMM_REPO:-/nix/yanjun/zts/alphabot2/lmm_repo}
HAB=${HABITAT_LAB:-/nix/yanjun/zts/lmm/home-robot/src/third_party/habitat-lab/habitat-lab}
ASSETS=${ALPHABOT2_ASSETS:-/nix/yanjun/zts/alphabot2/habitat_assets}
cp $REPO/configs/alphabot2_habitat/ovmm_alphabot2.yaml $HAB/habitat/config/benchmark/ovmm/ovmm_alphabot2.yaml
cp $REPO/configs/alphabot2_habitat/alphabot2.physics_config.json $ASSETS/alphabot2.physics_config.json
sha256sum $HAB/habitat/config/benchmark/ovmm/ovmm_alphabot2.yaml $ASSETS/alphabot2.physics_config.json
