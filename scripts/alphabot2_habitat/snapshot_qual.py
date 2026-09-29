"""Snapshot/restore qualification for AlphaBot 2 OVMM episodes (Habitat).

Per episode (the privileged expert provides a realistic mid-task state):
1. uninterrupted run: when the CARRY stage starts (object held by one hand) a
   snapshot S is captured; every executed JointMobileAction18 after S (sequence A,
   through navigation, placement, release and the object falling) is recorded with
   the state after each step (reference trajectory);
2. restore S -> replay A (twice): per-step divergence vs the reference, and between
   the two replays; OVMM place success equal?
3. siblings: restore S -> continuation B1 (turn left while holding), restore S ->
   B2 (drive back and release early); each twice. The state right after every
   restore must equal S exactly; each sibling must be repeatable.

    EPISODES=... OUT=... run_habitat.sh snapshot_qual.py
"""
import json, os, sys, time, traceback

import numpy as np

REPO = os.environ.get("LMM_REPO", "/nix/yanjun/zts/alphabot2/lmm_repo")
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import snapshot as SN  # noqa: E402
from lmm.embodiments.alphabot2.habitat.expert import OVMMExpert  # noqa: E402

OUT = os.environ.get("OUT", "/nix/yanjun/zts/alphabot2/habitat_runs/snapshot_qual")
os.makedirs(OUT, exist_ok=True)
env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=os.environ.get("SPLIT", "val"))
EPS = os.environ["EPISODES"].split(",")


def place_success():
    return float(env.get_metrics().get("ovmm_place_success", 0.0))


def run_seq(seq):
    traj = []
    for a in seq:
        HE.step18(env, a)
        traj.append(SN.state_vector(env))
    return traj


def max_div(t1, t2):
    d = [SN.divergence(a, b) for a, b in zip(t1, t2)]
    return {k: (max(x[k] for x in d) if k != "held_equal" else all(x[k] for x in d)) for k in d[0]} if d else None


with open(f"{OUT}/results.jsonl", "a") as fout:
    for eid in EPS:
        rec = {"episode_id": eid}
        t0 = time.time()
        try:
            HE.reset_to(env, eid)
            HE.fix_start(env)
            box = {"S": None, "A": [], "ref": [], "hand": None}
            orig = HE.step18

            def recording_step(env_, a):
                obs = orig(env_, a)
                if box["S"] is not None:
                    box["A"].append(np.asarray(a, np.float64).copy())
                    box["ref"].append(SN.state_vector(env_))
                return obs

            ex = None

            def on_step(cart, stage, rep):
                if stage == "carry" and box["S"] is None:
                    box["S"] = SN.capture(env, extra={"exec_q": {k: v.copy() for k, v in ex.ex.q.items()}})
                    box["S_state"] = SN.state_vector(env)
                    box["hand"] = ex.side
            HE.step18 = recording_step
            ex = OVMMExpert(env, on_step=on_step)
            try:
                ex.run()
                rec["expert_completed"] = True
            except Exception as e:
                rec["expert_completed"] = False
                rec["failure"] = repr(e)
            HE.step18 = orig
            rec["ref_place_success"] = place_success()
            if box["S"] is None:
                rec["skipped"] = "never reached CARRY"
                raise RuntimeError("no snapshot")
            rec["hand"], rec["continuation_steps"] = box["hand"], len(box["A"])
            t1 = time.time()
            reps = []
            for r in range(2):
                SN.restore(env, box["S"])
                rec[f"restore{r}_equals_S"] = SN.divergence(SN.state_vector(env), box["S_state"])
                reps.append(run_seq(box["A"]))
                rec[f"replay{r}_place_success"] = place_success()
            rec["restore_s"] = round((time.time() - t1) / 2, 3)
            rec["replay_vs_uninterrupted"] = max_div(reps[0], box["ref"])
            rec["replay_vs_replay"] = max_div(reps[0], reps[1])
            # siblings
            hold = box["A"][0].copy()
            hold[16] = hold[17] = 0.0
            b1 = [np.r_[hold[:16], 0.0, 0.5] for _ in range(20)]                           # turn left, holding
            rel = hold.copy()
            rel[7] = rel[15] = 0.0
            b2 = [np.r_[hold[:16], -0.2, 0.0] for _ in range(10)] + [np.r_[rel[:16], 0.0, 0.0] for _ in range(15)]  # back up, release
            sib = {}
            for name, seq in (("B1", b1), ("B2", b2)):
                runs = []
                for r in range(2):
                    SN.restore(env, box["S"])
                    sib[f"{name}_restore{r}_equals_S"] = SN.divergence(SN.state_vector(env), box["S_state"])
                    runs.append(run_seq(seq))
                sib[f"{name}_repeatable"] = max_div(runs[0], runs[1])
                sib[f"{name}_final_held"] = runs[0][-1]["held"]
            rec["siblings"] = sib
        except Exception as e:
            if "no snapshot" not in str(e):
                rec["engineering_error"] = repr(e)
                rec["tb"] = traceback.format_exc()[-1500:]
        rec["wall_s"] = round(time.time() - t0, 1)
        fout.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        fout.flush()
        print("SNAP", json.dumps({k: rec.get(k) for k in ("episode_id", "expert_completed", "hand", "continuation_steps",
                                                           "replay_vs_uninterrupted", "replay_vs_replay", "engineering_error")}),
              flush=True)
env.close()
