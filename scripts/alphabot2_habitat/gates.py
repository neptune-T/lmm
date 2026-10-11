"""Automatic gates between pipeline stages. One ###GATE### JSON line; exit 1 = BLOCKED.

    gates.py screen   NEW_SUMMARY COMMITTED_SUMMARY   full re-screen reproduces train 6331 / val 133 / 38 of 38
    gates.py navmesh  NAV_QUAL_DIR                    radius-0.38 val audit reproduces 7 goals-unreachable-at-reset
    gates.py smoke    SMOKE_JSON                      every runtime smoke check passes
    gates.py manip    MANIP_DIR                       >=1 complete forced-left, >=1 forced-right, >=3 auto successes
    gates.py snapshot SNAPSHOT_DIR                    tolerance gate (not bit-identical; see SNAP_TOL)
    gates.py pilot    COLLECT_DIR PILOT_LIST          10-20 complete successes, each: fresh-reset replay,
                                                      timing, no privileged leakage; both arms; >1 scene
"""
import glob, json, os, re, sys
from collections import defaultdict

import numpy as np


def files(d, pat="results*.jsonl"):
    """glob + explicit probing of the names our writers use: bosfs directory listings lag behind file
    creation (stat/open see a new file, readdir may not for a while)."""
    stem = pat.split("*")[0]
    names = [f"{stem}.jsonl", f"{stem}_left.jsonl", f"{stem}_right.jsonl"] + \
            [f"{stem}_w{i:02d}.jsonl" for i in range(100)] + [f"{stem}{i:02d}.jsonl" for i in range(100)]
    return sorted(set(glob.glob(f"{d}/{pat}")) | {f"{d}/{n}" for n in names if os.path.exists(f"{d}/{n}")})


def rows(d, pat="results*.jsonl"):
    out = []
    for f in files(d, pat):
        for line in open(f):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def emit(name, ok, **kw):
    print("###GATE###" + json.dumps({"gate": name, "pass": bool(ok), **kw}, default=str), flush=True)
    sys.exit(0 if ok else 1)


def complete_success(r):
    return (not r.get("engineering_error") and bool(r.get("expert_completed"))
            and float((r.get("ovmm") or {}).get("ovmm_place_success") or 0) == 1.0)


kind = sys.argv[1]

if kind == "screen":
    new, old = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    flat = lambda d: {k: v for k, v in d.items() if not isinstance(v, dict)}
    t, v = new["train"], new["val"]
    ok = (t["feasible"] == 6331 and v["feasible"] == 133 and t["feasible_scenes"] == 38 and t["scenes"] == 38
          and all(flat(new[s]) == flat(old[s]) for s in ("train", "val")))
    emit("screen", ok, train=flat(t), val=flat(v), committed={s: flat(old[s]) for s in ("train", "val")})

if kind == "navmesh":
    # gates.py navmesh DOC_SETTINGS_DIR MAINLINE_DIR
    # 1) the documented audit (docs/ALPHABOT2_OVMM_EMBODIMENT.md §5, radius 0.38 row) was measured with
    #    navmesh height 1.7: rerun at r0.38/h1.7 it must reproduce 24 episodes / 7 goals unreachable at reset;
    # 2) mainline config (r0.38/h1.5, robot 1.434 m tall): its unreachable set must be a subset of (1)'s
    #    (a lower agent height can only open the navmesh); counts are recorded as the new reference.
    import glob as _g
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from summarize_nav import summarize

    def unreachable(d):
        return sorted({json.loads(l)["episode_id"] for f in _g.glob(f"{d}/results_w*.jsonl") for l in open(f)
                       if "None of the goals" in (json.loads(l).get("engineering_error") or "")})
    doc, main = summarize(sys.argv[2]), summarize(sys.argv[3])
    ud, um = unreachable(sys.argv[2]), unreachable(sys.argv[3])
    ok = (doc["episodes"] == 24 and doc["reset_goals_unreachable"] == 7 and not doc["other_errors"]
          and main["episodes"] == 24 and not main["other_errors"] and set(um) <= set(ud))
    pick = lambda s: {k: s[k] for k in ("episodes", "scenes", "reset_goals_unreachable", "both_legs_arrived",
                                        "both_legs_collision_free")}
    emit("navmesh", ok, documented_settings_r038_h17=pick(doc), unreachable_h17=ud,
         mainline_r038_h15=pick(main), unreachable_h15=um,
         documented={"episodes": 24, "reset_goals_unreachable": 7, "both_legs_arrived": 12, "both_legs_collision_free": 11})

if kind == "embodiment":
    # gates.py embodiment EMB_DIR GEOMETRIC_SCREEN_JSON UNITS_OUT_JSON
    # Measurement + completeness/determinism gate (no outcome filtering, no ratio): every geometry-feasible
    # episode screened, no engineering errors, seeded resets deterministic. Writes the AlphaBot-feasible
    # units (episode, replica): geometry AND reset-reachable AND some arm pick-feasible AND place-feasible.
    from collections import Counter
    er = {}
    for f in files(sys.argv[2], "emb_w*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            er[r["episode_id"]] = r
    geo = json.load(open(sys.argv[3]))
    gf = [r for r in geo if r["feasible"]]
    missing = [r["episode_id"] for r in gf if r["episode_id"] not in er]
    eng = [k for k, r in er.items() if r.get("engineering_error")]
    reps = [u for r in er.values() for u in r.get("replicas", [])]
    nondet = [(k, u["replica"]) for k, r in er.items() for u in r.get("replicas", []) if u.get("repeat_start_pose_equal") is False]
    checked = sum(1 for u in reps if "repeat_start_pose_equal" in u)
    units = []
    for r in er.values():
        for u in r.get("replicas", []):
            if u.get("reachable") and r.get("feasible_arms"):
                units.append({"unit": f"{r['episode_id']}#{u['replica']}", "episode_id": r["episode_id"],
                              "replica": u["replica"], "scene": r["scene"], "object": r.get("object"),
                              "feasible_arms": r["feasible_arms"]})
    json.dump(units, open(sys.argv[4], "w"))
    ok_ep = [r for r in er.values() if not r.get("engineering_error")]
    reach_ep = [r for r in ok_ep if any(u.get("reachable") for u in r["replicas"])]
    arm = lambda side, key: sum(1 for r in reach_ep if r["arms"].get(side, {}).get(key))
    both = lambda side: sum(1 for r in reach_ep if side in r.get("feasible_arms", []))
    feas_ep = [r for r in reach_ep if r.get("feasible_arms")]
    emit("embodiment", not missing and not eng and not nondet and checked > 0,
         funnel_episodes={"full_ovmm_train": len(geo), "geometry_feasible": len(gf), "reset_reachable_any_replica": len(reach_ep),
                          "pick_feasible_left": arm("left", "pick"), "pick_feasible_right": arm("right", "pick"),
                          "pick_and_place_left": both("left"), "pick_and_place_right": both("right"),
                          "alphabot_feasible": len(feas_ep)},
         scenes={"full_ovmm_train": len({r["scene"] for r in geo}), "geometry_feasible": len({r["scene"] for r in gf}),
                 "alphabot_feasible": len({r["scene"] for r in feas_ep})},
         objects_alphabot_feasible=len({r.get("object") for r in feas_ep}),
         feasible_arm_sets=dict(Counter("+".join(sorted(r["feasible_arms"])) for r in feas_ep)),
         replicas_reachable=f"{sum(1 for u in reps if u.get('reachable'))}/{len(reps)}", units=len(units),
         missing=missing[:10], engineering_errors=eng[:10], determinism_checked=checked, nondeterministic=nondet[:10])

if kind == "smoke":
    r = json.load(open(sys.argv[2]))
    failed = [k for k, v in r.items() if isinstance(v, dict) and "pass" in v and not v["pass"]]
    emit("smoke", not failed and len(r) >= 15, checks=len(r), failed=failed, episode=r.get("01_load", {}).get("episode"))

if kind == "manip":
    rs = rows(sys.argv[2])
    by = defaultdict(lambda: {"n": 0, "complete_success": 0, "episodes_ok": [], "eng_err": 0})
    for r in rs:
        m = by[r["mode"]]
        m["n"] += 1
        m["eng_err"] += int(bool(r.get("engineering_error")))
        forced_ok = r["mode"] == "auto" or r.get("arm") == r["mode"]
        if complete_success(r) and forced_ok:
            m["complete_success"] += 1
            m["episodes_ok"].append(r["episode_id"])
    ok = by["left"]["complete_success"] >= 1 and by["right"]["complete_success"] >= 1 and by["auto"]["complete_success"] >= 3
    emit("manip", ok, modes=dict(by),
         auto_arms={a: sum(1 for r in rs if r["mode"] == "auto" and complete_success(r) and r.get("arm") == a)
                    for a in ("left", "right")})

if kind == "snapshot":
    # Tolerances from the measured envelope of the Isaac contact-rich audit
    # (docs/ALPHABOT2_SNAPSHOT_RESTORE_AUDIT.md, commit 2713b5f: restore vs uninterrupted <= 0.094 rad joints,
    # 0.65 mm base, 23.2 mm held object; branches from one snapshot identical; readback exact).
    SNAP_TOL = {"readback": {"root_max_abs": 1e-6, "proprio_max_abs": 1e-6, "objects_max_abs": 1e-3},
                "replay_vs_uninterrupted": {"root_max_abs": 5e-3, "proprio_max_abs": 0.1, "objects_max_abs": 0.025},
                "branch_repeat": {"root_max_abs": 5e-3, "proprio_max_abs": 0.1, "objects_max_abs": 0.025}}

    def within(d, tol):
        return bool(d["held_equal"]) and all(d[k] <= v for k, v in tol.items())

    rs = [r for r in rows(sys.argv[2]) if "skipped" not in r]
    per, bad = [], []
    for r in rs:
        if r.get("engineering_error"):
            bad.append((r["episode_id"], "engineering_error", r["engineering_error"][:200]))
            continue
        sib = r["siblings"]
        checks = {
            "readback_restore0": within(r["restore0_equals_S"], SNAP_TOL["readback"]),
            "readback_restore1": within(r["restore1_equals_S"], SNAP_TOL["readback"]),
            "grasp_state_after_restore": all(r[f"restore{i}_equals_S"]["held_equal"] for i in (0, 1)),
            "replay_vs_uninterrupted": within(r["replay_vs_uninterrupted"], SNAP_TOL["replay_vs_uninterrupted"]),
            "replay_vs_replay": within(r["replay_vs_replay"], SNAP_TOL["branch_repeat"]),
            "siblings_readback": all(within(sib[f"{b}_restore{i}_equals_S"], SNAP_TOL["readback"])
                                     for b in ("B1", "B2") for i in (0, 1)),
            "siblings_consistent": all(within(sib[f"{b}_repeatable"], SNAP_TOL["branch_repeat"]) for b in ("B1", "B2")),
            "task_outcome": r["replay0_place_success"] == r["ref_place_success"] == r["replay1_place_success"],
        }
        per.append({"episode_id": r["episode_id"], "hand": r.get("hand"), **checks,
                    "replay_vs_uninterrupted": r["replay_vs_uninterrupted"], "replay_vs_replay": r["replay_vs_replay"]})
        if not all(checks.values()):
            bad.append((r["episode_id"], [k for k, v in checks.items() if not v]))
    exact = all(max(p["replay_vs_replay"][k] for k in p["replay_vs_replay"] if k.endswith("_max_abs")) == 0.0 for p in per)
    emit("snapshot", len(per) >= 3 and not bad, n=len(per), problems=bad, tolerances=SNAP_TOL,
         verdict=("NOT_READY" if bad else "EXACT_BRANCHING" if exact else "USABLE_WITH_TOLERANCE"), episodes=per)

if kind == "pilot":
    out_dir = sys.argv[2]
    pil = set(open(sys.argv[3]).read().replace(",", "\n").split())
    rs = [r for r in rows(out_dir) if str(r["episode_id"]) in pil]
    succ = [r for r in rs if r.get("success") and r.get("accepted")]          # replay-accepted successes
    replay_rejected = [r["episode_id"] for r in rs if r.get("success") and not r.get("accepted")]
    CAMS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
    LOWDIM = {"proprio", "action", "joint18", "sim_time"}
    ARM_WORDS = re.compile(r"\b(left|right|arm|hand|gripper|stage|grasp)\b", re.I)
    per, bad = [], []
    for r in succ:
        eid, d = str(r["episode_id"]), r.get("dir") or f"{out_dir}/episodes/{r['episode_id']}"
        c = {"replay_fresh_reset": bool(r.get("accepted")) and bool(r["replay"].get("start_fix_equal"))}   # tolerance rule (pilot.py)
        try:
            im, lo = np.load(f"{d}/images.npz"), np.load(f"{d}/lowdim.npz")
            meta = json.load(open(f"{d}/meta.json"))
            T = lo["action"].shape[0]
            dt = np.diff(lo["sim_time"])
            c["timing_aligned"] = (all(im[k].shape[0] == T for k in CAMS) and lo["proprio"].shape[0] == T
                                   and lo["joint18"].shape[0] == T and lo["sim_time"].shape[0] == T
                                   and T == r["steps"] and bool(np.all(np.abs(dt - 0.1) < 1e-6)))
            c["no_privileged_leakage"] = (set(im.files) == set(CAMS) and set(lo.files) == LOWDIM
                                          and all(im[k].dtype == np.uint8 and im[k].ndim == 4 and im[k].shape[-1] == 3 for k in CAMS)
                                          and lo["proprio"].shape[1] == 16 and lo["action"].shape[1] == 18
                                          and meta["policy_inputs"] == list(CAMS) + ["proprio16", "instruction"]
                                          and not ARM_WORDS.search(meta["instruction"]))
        except Exception as e:                                           # missing/corrupt files fail the episode
            c["files_readable"] = False
            c["error"] = repr(e)[:200]
        per.append({"episode_id": eid, "arm": r.get("arm"), "scene": r.get("scene"), "steps": r.get("steps"), **c})
        if not all(v for k, v in c.items() if k != "error"):
            bad.append(eid)
    good = [p for p in per if p["episode_id"] not in bad]
    arms = {a: sum(p["arm"] == a for p in good) for a in ("left", "right")}
    ok = (10 <= len(good) <= 20 and not bad and arms["left"] >= 1 and arms["right"] >= 1
          and len({p["scene"] for p in good}) > 1)
    emit("pilot", ok, processed=len(rs), complete_successes=len(good), failing_checks=bad, arms=arms,
         replay_rejected_successes=replay_rejected,
         replay_reject_fraction=round(len(replay_rejected) / max(1, len(replay_rejected) + len(succ)), 3),
         scenes=len({p["scene"] for p in good}), engineering_errors=[r["episode_id"] for r in rs if r.get("engineering_error")],
         episodes=per)
if kind == "dataqc":
    # gates.py dataqc COLLECT_DIR UNITS_JSON MIN_ACCEPTED MIN_ARM_SHARE MIN_SCENE_COVERAGE [MIN_SCENES]
    # Hard data gate before SFT (presets live in configs/alphabot2_pi05/schedule.yaml): enough accepted
    # trajectories; EVERY accepted trajectory intact, 10 Hz-aligned and leakage-free; no determinism break
    # anywhere (success but fresh-reset replay mismatch); both arms and the feasible scenes covered.
    import zipfile, ast
    out_dir, units = sys.argv[2], json.load(open(sys.argv[3]))
    n_min, arm_min, cov_min = int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
    scenes_min = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    rs = rows(out_dir)
    acc = [r for r in rs if r.get("accepted")]
    CAMS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
    ARM_WORDS = re.compile(r"\b(left|right|arm|hand|gripper|stage|grasp)\b", re.I)

    def npz_shapes(path):
        out = {}
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                with z.open(n) as f:
                    v = np.lib.format.read_magic(f)
                    rd = np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0
                    shape, _, dtype = rd(f)
                    out[n[:-4]] = (tuple(shape), str(dtype))
        return out

    bad = []
    for r in acc:
        d = r.get("dir")
        try:
            im = npz_shapes(f"{d}/images.npz")
            lo = np.load(f"{d}/lowdim.npz")
            meta = json.load(open(f"{d}/meta.json"))
            T = int(r["steps"])
            ok = (set(im) == set(CAMS) and all(im[c][0] == (T, 240, 320, 3) and im[c][1] == "uint8" for c in CAMS)
                  and set(lo.files) == {"proprio", "action", "joint18", "sim_time"}
                  and lo["proprio"].shape == (T, 16) and lo["action"].shape == (T, 18) and lo["joint18"].shape == (T, 18)
                  and lo["sim_time"].shape == (T,) and bool(np.all(np.abs(np.diff(lo["sim_time"]) - 0.1) < 1e-6))
                  and os.path.exists(f"{d}/metadata.npz")
                  and meta["policy_inputs"] == list(CAMS) + ["proprio16", "instruction"]
                  and not ARM_WORDS.search(meta["instruction"]))
            if not ok:
                bad.append(r["episode_id"])
        except Exception as e:
            bad.append(f"{r['episode_id']}: {repr(e)[:80]}")
    mismatch = [r["episode_id"] for r in rs if r.get("success") and not r.get("accepted")]
    max_rej = float(os.environ.get("MAX_REPLAY_REJECT_FRACTION", "0.2"))
    rej_frac = len(mismatch) / max(1, len(mismatch) + len(acc))
    arms = {a: sum(r.get("arm") == a for r in acc) for a in ("left", "right")}
    feas_scenes = {u["scene"] for u in units}
    cov = len({r.get("scene") for r in acc} & feas_scenes) / max(1, len(feas_scenes))
    share = {a: (arms[a] / len(acc) if acc else 0.0) for a in arms}
    n_scenes = len({r.get("scene") for r in acc})
    ok = (len(acc) >= n_min and not bad and rej_frac <= max_rej and all(v >= arm_min for v in share.values())
          and cov >= cov_min and n_scenes >= scenes_min)
    emit("dataqc", ok, scenes=n_scenes, min_scenes=scenes_min, accepted=len(acc), processed=len(rs), min_accepted=n_min, integrity_failures=bad[:20],
         n_integrity_failures=len(bad), success_but_replay_mismatch=mismatch[:20], replay_reject_fraction=round(rej_frac, 3),
         max_replay_reject_fraction=max_rej, arm_share=share, min_arm_share=arm_min,
         scene_coverage=round(cov, 3), feasible_scenes=len(feas_scenes), min_scene_coverage=cov_min,
         objects=len({r.get("object") for r in acc}), accepted_steps=sum(int(r.get("steps") or 0) for r in acc))

emit(kind, False, error="unknown gate")
