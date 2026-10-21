"""Snapshot/restore qualification for AlphaBot 2 OVMM episodes (Habitat).

Per episode (the privileged expert provides a realistic mid-task state):
1. uninterrupted run: when the CARRY stage starts (object held by one hand) a
   snapshot S is captured; every executed JointMobileAction18 after S (sequence A,
   through navigation, placement, release and the object falling) is recorded with
   the state after each step (reference trajectory);
2. reconstruct S in a NEW simulator via the ordered action/contact prefix -> replay A (twice): per-step divergence vs the reference, and between
   the two replays; OVMM place success equal?
3. siblings: restore S -> continuation B1 (turn left while holding), restore S ->
   B2 (drive back and release early); each twice. The state right after every
   reconstruction must equal S exactly; each sibling must be repeatable.

The in-place SN.restore path is NOT qualified by this script. Episode reset alone
retains hidden simulator state: measured release-branch drift8.90deg in15021.
Full simulator recreation plus ordered event replay removed that drift in the
single-episode diagnostic; the same gates qualify the entire selected set here.

    EPISODES=... OUT=... run_habitat.sh snapshot_qual.py
"""
import json, os, sys, time, traceback

import numpy as np

REPO = os.environ.get("LMM_REPO", "/workspace/lmm")
sys.path.insert(0, REPO)
from lmm.embodiments.alphabot2.habitat import env as HE  # noqa: E402
from lmm.embodiments.alphabot2.habitat import snapshot as SN  # noqa: E402
from lmm.embodiments.alphabot2.habitat import nav as NAV  # noqa: E402
from lmm.embodiments.alphabot2.replay_events import replay_events  # noqa: E402
from lmm.embodiments.alphabot2.habitat.expert import OVMMExpert  # noqa: E402

OUT = os.environ.get("OUT", "/workspace/runtime/runs/snapshot_qual")
os.makedirs(OUT, exist_ok=True)
env = None
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
    return SN.max_divergence([SN.divergence(a, b) for a, b in zip(t1, t2)])


with open(f"{OUT}/results.jsonl", "a") as fout:
    for eid in EPS:
        rec = {"episode_id": eid, "restore_mode": "fresh_simulator_event_prefix"}
        t0 = time.time()
        try:
            if env is not None:
                env.close()
            env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=os.environ.get("SPLIT", "val"))
            HE.reset_to(env, eid)
            HE.fix_start(env)
            box = {"S": None, "A": [], "ref": [], "hand": None}
            orig = HE.step18
            orig_contacts = NAV.scene_contacts
            prefix = []

            def recording_contacts(env_):
                prefix.append({"kind": "contacts"})
                return orig_contacts(env_)

            def recording_step(env_, a):
                prefix.append({"kind": "step", "action": np.asarray(a).tolist()})
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
                    box["prefix_end"] = len(prefix)
                    box["hand"] = ex.side
            HE.step18 = recording_step
            NAV.scene_contacts = recording_contacts
            ex = OVMMExpert(env, on_step=on_step)
            try:
                ex.run()
                rec["expert_completed"] = True
            except Exception as e:
                rec["expert_completed"] = False
                rec["failure"] = repr(e)
            HE.step18 = orig
            NAV.scene_contacts = orig_contacts
            rec["ref_place_success"] = place_success()
            if box["S"] is None:
                rec["skipped"] = "never reached CARRY"
                raise RuntimeError("no snapshot")
            rec["hand"], rec["continuation_steps"] = box["hand"], len(box["A"])
            with open(f"{OUT}/prefix_{eid}.json", "w") as f:
                json.dump({"episode_id": eid, "prefix": prefix[:box["prefix_end"]]}, f)
            def restore_root():
                global env
                env.close()
                env = HE.make_env(gpu=int(os.environ.get("GPU", "0")), split=os.environ.get("SPLIT", "val"))
                HE.reset_to(env, eid)
                HE.fix_start(env)
                replay_events(prefix[:box["prefix_end"]],
                              lambda a: orig(env, a), lambda: orig_contacts(env))
            t1 = time.time()
            reps = []
            for r in range(2):
                restore_root()
                rec[f"restore{r}_equals_S"] = SN.divergence(SN.state_vector(env), box["S_state"])
                trajectory = []
                replay_events(prefix[box["prefix_end"]:], lambda a: orig(env, a),
                              lambda: orig_contacts(env),
                              lambda: trajectory.append(SN.state_vector(env)))
                if len(trajectory) != len(box["ref"]):
                    raise RuntimeError("Replay/reference step count mismatch")
                reps.append(trajectory)
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
                    restore_root()
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
