# A800 execution host

This is the non-secret operational record for the restored OVMM host. Never
commit passwords, private keys, access tokens, or `authorized_keys` contents.

## Access

- Target: `root@180.76.138.216`, SSH port `12`
- Jump host alias: `jumper` (`yanjun@100.74.253.40`, Tailscale)
- Working route: enter the Windows jump host, then use its dedicated
  `C:/Users/yanjun/.ssh/id_ed25519_a800` identity for the target.
- The workstation's ordinary key is not the target identity. Direct
  `ProxyJump` with that key can be rejected even when the server is healthy.
- Do not modify files on the jump host and do not scan ports.

Example read-only health check:

```bash
ssh -tt jumper \
  'ssh -tt -o IdentitiesOnly=yes \
    -i C:/Users/yanjun/.ssh/id_ed25519_a800 \
    -p 12 root@180.76.138.216 \
    "hostname; nvidia-smi"'
```

## Hardware and storage (verified 2026-09-20)

- Hostname: `aibox-r8966c444da7-676d647d8b-dpddc`
- GPUs: 4 × NVIDIA A800-SXM4-80GB
- `/nix`: 3.5 TB total, approximately 2.5 TB free at verification time
- GPU 1-3 were idle. GPU 0 had `/share/cys/occupy.py` using about 1 GB; do not
  stop it without its owner's permission.

## Canonical server paths

- Workspace root: `/nix/yanjun/zts/lmm`
- Canonical repository: `/nix/yanjun/zts/lmm/lmm`
- Legacy companion: `/nix/yanjun/zts/lmm/lmm_rollout_project`
- HomeRobot: `/nix/yanjun/zts/lmm/home-robot`
- OpenPI: `/nix/yanjun/zts/lmm/codebase/openpi`
- Habitat Python: `/nix/yanjun/zts/lmm/envs/home_robot39/bin/python`
- OpenPI Python: `/nix/yanjun/zts/lmm/envs/openpi311/bin/python`
- Recovered checkpoints: `/nix/yanjun/zts/lmm/checkpoints`
- HSSD: `/nix/yanjun/zts/lmm/home-robot/data/hssd-hab`

## Verified OVMM dual-arm state

The Trossen Mobile AI overlay is installed alongside, not over, Stretch. On
2026-09-20 a real GPU1 smoke test completed:

1. Habitat dataset and simulator initialization;
2. episode 743 reset;
3. head RGB/depth and both wrist RGB renders;
4. finite 14D dual-arm proprioception;
5. action space construction (12 arm + 2 gripper + 2 base = 16D);
6. zero-velocity environment step and clean close.

The implementation currently lives in the legacy companion under
`habitat_patch/trossen_mobile_ai/` and `scripts/rollout/`. It still needs to be
bound behind canonical `BehaviorEnv.make()` before an end-to-end policy
experiment can run from this repository.

## Checkpoint constraint

Recovered q20 is a Stretch checkpoint: its norm statistics are 10D state and
10D action. It must not be presented as a Trossen dual-arm policy. The server
contains OpenPI mobile profiles with 14D state / 16D action, but an exact
Trossen-Mobile-AI OVMM checkpoint or freshly computed task-specific norm
statistics is still required.
