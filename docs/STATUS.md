# LMM / AlphaBot 2 OVMM: status, lessons and recovery plan (2026-10-19)

> **2026-10-28 current recovery:** read [recovery_20261028.md](recovery_20261028.md)
> first. The previous instance stopped after an HF staging ENOSPC failure killed
> training. New endpoint is port5140; all new runtime writes belong under
> `/share/personal/zhangtianshan/lmm`, persistent snapshots under `/workspace/zts/lmm`.
> Recover only the verified COMPLETE step1000 checkpoint; step2000 is incomplete.
> **Latest user correction:** no further `/workspace` writes. Use the CFS path
> above for code/data/checkpoints; HF upload runs after training, not as a launch gate.

> **2026-10-27 recovery update:** read [recovery_20261027.md](recovery_20261027.md)
> first. The original converted v2 train/val data has been recovered from CFS,
> fully decoded and backed up with verified HF hashes. Dual-A800 B SFT
> reconstruction from the public base is authorized. Original v2-B weights
> and raw simulator episodes remain unavailable. The host, GPU assignment and
> active run paths in older updates are historical, not current launch settings.

> **2026-10-20 recovery update:** the historical status below is not live server state.
> Read [recovery_20261020.md](recovery_20261020.md) first. New host access is verified; all GPUs
> are occupied, and the user selected GPU7 only. HF contains code/results, not v2 weights.
> Reconstruction is authorized; no automatic frontier/RL. Do not reuse old launch commands,
> keep-alive rules or template loss weights without the new recovery gates.

## 1. Where the project stands

| Item | State |
|---|---|
| Research goal | Frontier Discovery + Rollout Scaling + Online RL on top of a baseline with reproducible full-task success. NAV no longer has to be learned from RGB. |
| Frozen baseline | **PA-OG + base lock + v2-B** (planner-assisted, oracle goal; π0.5 does all manipulation through its 18-D interface; observable phase switches). Contract: `docs/v2/pa_baseline.md`, code `scripts/alphabot2_habitat/pa_eval.py`. |
| Baseline results (v2-B) | train units 15/60 (25 %, seeds 0–2), held-out episodes 3/40, OVMM-val 0/30. 40/41 recorded successes reproduce by open-loop replay (16 episodes). From expert arrival: place → full ≈ 46 %, grasp → carry-ready ≈ 24 % (the bottleneck). |
| M-SFT v1 (manipulation-only continuation) | train 15/40 vs v2-B 10/40 (paired 8 vs 3, p = 0.23, trend only), held-out 1/20, val 1/30 (not replay-reproducible). **Not adopted**; v2-B stays the frozen baseline. |
| Determinism | batched serving is not bitwise reproducible; `--max-batch 1` is (two runs bitwise identical). |
| Manipulation frontier pilot | code + protocol ready (`docs/v2/frontier_pilot.md`, `frontier_pa.py`, `frontier_plan.py`, `frontier_analysis.py`, `frontier_chain.sh`), **not run**: the LMM host was lost on 2026-10-19 ~20:45 before the first rollout. |
| Server data | lost with the host restart (per the user): v2-B / M-SFT checkpoints, v2 datasets, collected expert episodes (3 444 accepted), all run traces and videos. Survives only what was copied off the host (below). |
| Off-host copies | code: GitHub `neptune-T/lmm` + HF bundle; result data + report + filmstrip: `docs/v2/results/`; this file; claude.ai report artifact. |

Result numbers above are reproducible from `docs/v2/results/pa_report_data_2026-10-19.json` (all pass funnels, successes,
replay records, arrival-diagnostic summaries) with `scripts/alphabot2_habitat/pa_report_html.py`.

## 2. Pitfalls hit this week and the rule that prevents each

| # | What happened | Rule from now on |
|---|---|---|
| 1 | **No off-host backup existed.** `hf_backup.py` was written but never ran (the server token was fine-grained and could not create the repo); a restart then took every large file. | After **every** task (data collection, conversion, training, evaluation, code change) run `scripts/hf_sync.py` to the private HF dataset repo `Neptune-T/lmm-alphabot2-ovmm-runs`; a task counts as done only when its verification exits 0. Code also goes to GitHub. |
| 2 | Container OOM restart: the 481 GB cgroup limit counts `/dev/shm`; ~40 sim workers + 177 GB of tmpfs checkpoints exceeded it. | Budget rss + shmem before launching (≤ ~24 sim workers per pod); big checkpoints never live only in `/dev/shm`; run the memory log (`memlog.sh`, warning > 80 %). |
| 3 | Host stopped when every GPU went idle (the OOM restart had killed the GPU-3 keep-alive; I then stopped the last servers). | After any restart check the keep-alive first; never leave all GPUs idle; ask before stopping the last GPU job. |
| 4 | Killing a tmux pane's shell SIGHUP'd the workers in that pane. | Launch workers with `setsid … < /dev/null &`; never kill a pane shell to stop a runner. |
| 5 | A runner's final `pkill` would have killed policy servers shared with other jobs. | One server set per runner, or no `pkill` in shared runners. |
| 6 | bosfs: files being written read back as "Invalid argument" / missing from `ls`; no symlinks. | Live logs in `/dev/shm`; results written atomically (build in `/dev/shm`, then copy). |
| 7 | Closed-loop re-runs were not reproducible (batched fp32 serving). | Any rollout that must be re-executed (roots, frontier, RL) uses `--max-batch 1` servers and asserts `batch_size == 1`. |
| 8 | Open-loop replay of joint actions alone missed side effects (render → `robot.update()` joint clamping; contact queries → Bullet contact cache): one success did not replay. | Exact restore replays the full side-effect event log (step / render / contact query) — `frontier_pa.py`. |
| 9 | `WORKER::NWORKERS` slicing of a periodic job list left workers with only fast jobs (4 h tail). | Shuffle / balance job lists; workers skip jobs done by any worker file (union done-set). |
| 10 | Stale checkpoint mirror (size-only signature) — evaluations nearly used step-1000 weights. | Verify checkpoints by sha256 before every eval; mirror signatures include mtimes. |
| 11 | Local scratchpad wiped: report generator and helper scripts lost. | Anything worth keeping goes into the repo and is committed + pushed immediately. |
| 12 | A pre-registered rule written from 7 rows (M-SFT place ≥ 25 %) turned out uninformative. | Decision rules are relative to the paired baseline and set after a minimum sample, or explicitly marked provisional. |
| 13 | `Neptune-T/LMM-GRPO-Handoff` is public (flagged 09-14, still public). | Owner to flip it private; backups only ever go to private repos (hf_sync refuses public ones). |
| 14 | Older traps still valid: no C compiler → `TORCHDYNAMO_DISABLE=1`; bf16-only updates vanish (fp32 master params); lerobot 0.6.1 quantile stats wrong (exact_stats.py); lerobot streaming encoder drops frames (converter aborts on drops). | Keep. |

## 3. Recovery plan (optimal order when a host is available)

1. **Inventory before touching anything**: what survived on `/workspace` and CFS (`du`, sha256 of the v2-B mirror
   `sft_mirror/v2B/ckpt_step0010000/policy/model.safetensors` = 405f2d5d…). Back up whatever survived to HF first.
2. **Host hygiene**: keep-alive running (owner), write token in `/root/.config/hf_token_write` (mode 600, never echoed),
   tmux session `lmm`, `memlog.sh`.
3. **Code**: clone from GitHub (or `git clone` the HF bundle); check out the latest commit.
4. **Runtime env** (`home_robot39`, habitat-lab b727ca9 + `configs/alphabot2_habitat/habitat_dual_hand.patch`, pi0.5 env
   with lerobot 0.6.1, torch 2.8 cu126): rebuild per `scripts/alphabot2_pi05/build_pi05_env.sh` and
   `scripts/alphabot2_habitat/run_habitat.sh`; **archive the built env tarballs + lockfiles to HF immediately**.
5. **Assets / data**: public AlphaBot pi0.5 checkpoint, OVMM/HSSD data, AlphaBot assets (`fetch_assets.py`); back up the
   staged asset bundle to HF.
6. **Model state**, if the v2-B weights are gone: regenerate in the original order (expert collection on the
   AlphaBot-feasible train units → LeRobot conversion with exact stats and video gate → v2-B SFT, `configs/alphabot2_pi05/sft_v2B.yaml`, 10k steps), backing each stage up to HF
   (episodes in shards, dataset, every checkpoint `best`/`last`). Then re-qualify: PA-OG + base lock on units_train20
   must again give a non-zero, replay-verified success before any new experiment.
7. **Then** the manipulation frontier pilot exactly as pre-registered (`frontier_chain.sh`, one command; it restores
   v2-B from the mirror if needed and refuses a sha256 mismatch), followed by `hf_sync.py` of its outputs.
