# Public AlphaBot π0.5 checkpoint — analysis and runtime scripts

Run in a Python 3.12 venv with `lerobot[pi,dataset]==0.6.1`, torch 2.8.0+cu126,
`av`, `sentencepiece` (the A800 setup is described in
`docs/ALPHABOT2_PI05_EXECUTION_CONTRACT.md`). Paths point at the host
workspace `/nix/yanjun/zts/alphabot2/hf` (checkpoint and dataset pinned by revision).

| Script | Establishes |
|---|---|
| `pi05_data_audit.py` | fps, tasks, quaternion norms/hemisphere, gripper polarity, first naive FK fit |
| `pi05_joint_conv.py` | frame-free joint-convention search (all frames; shared helpers `fk_batch`) |
| `pi05_disc.py` | limit-constrained discrete sign/offset search |
| `pi05_hyp.py`, `pi05_robust.py`, `pi05_refine.py` | robust frame+tool fits per convention hypothesis |
| `pi05_final_fit.py` | lag scan; shared vs per-arm base frame (source of the constants in `pi05_contract.py`) |
| `pi05_frames.py` | decodes training frames to identify what each camera key shows |
| `pi05_runtime.py` | loads the checkpoint exactly as trained (only the tokenizer source differs) |
| `pi05_offline_check.py`, `pi05_diag.py` | executes the checkpoint on training frames; input-sensitivity table |
| `pi05_fkbase.py` | how far recorded targets lead the observed arm |
| `pi05_server.py` | serves the checkpoint on 127.0.0.1 (numpy-version-neutral array codec) for the Isaac client |
| `pilot_to_lerobot.py` | pilot episodes -> local LeRobot v3 dataset (checkpoint state convention, 18-D action) |
| `sft_train.py` | **prepared, not launched**: lerobot-train + public-tokenizer shim, for `configs/alphabot2_pi05_sft_pilot.json` |
| `sft_dryrun.py` | SFT readiness: one forward loss, no backward/optimizer/checkpoint |

Outputs: `artifacts/alphabot2_pi05_contract_evidence.json`.
