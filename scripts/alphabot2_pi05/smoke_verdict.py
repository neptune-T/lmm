"""Top-level verdict of a v2 smoke run (stdlib only): every stage must pass before the full run starts.

  gates     gate_A / gate_B interface_gate.json: top-level pass and every check true
  sweep     each of w = 1, 2, 4: SFT DONE with finite losses; audit.json (v2-val navigation block present);
            closed_loop.json with >= 1 traced unit; decision.json with a chosen weight
  ab_train  v2smokeA / v2smokeB: DONE, finite train losses, best checkpoint saved; B ran with history inputs
  server_consistency  consistency_A / _B: fp32 server == in-process (bit-exact), processors == train stats
  eval      eval_A / eval_B train_cl summary: eval_count >= 1, engineering_errors == 0; server READY

    python3 smoke_verdict.py --root RUNS/v2smoke --sft /dev/shm/lmm_runtime/sft --out VERDICT.json
"""
import argparse, json, math, os


def finite_log(d):
    f = os.path.join(d, "train_log.jsonl")
    if not os.path.exists(f):
        return False
    rows = [json.loads(l) for l in open(f) if l.strip()]
    tr = [r for r in rows if "loss" in r and "val_loss" not in r]
    return bool(tr) and all(math.isfinite(r["loss"]) for r in tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sft", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R, S = a.root, a.sft
    v = {}
    g = {}
    for x in ("A", "B"):
        try:
            j = json.load(open(f"{R}/gate_{x}/interface_gate.json"))
            g[x] = {"pass": j["pass"] is True and all(j["checks"].values()), "failed": [k for k, c in j["checks"].items() if not c]}
        except (OSError, ValueError, KeyError) as e:
            g[x] = {"pass": False, "error": repr(e)}
    v["gates"] = {"pass": all(x["pass"] for x in g.values()), **g}
    sw = {}
    for w in ("1", "2", "4"):
        d = f"{S}/v2smokesweep_w{w}"
        e = {"sft_done": os.path.exists(f"{d}/DONE"), "finite": finite_log(d)}
        try:
            au = json.load(open(f"{R}/sweep/w{w}/audit.json"))
            e["audit_nav"] = "navigation" in au.get("val", {})
        except (OSError, ValueError):
            e["audit_nav"] = False
        try:
            e["closed_loop_units"] = len(json.load(open(f"{R}/sweep/w{w}/closed_loop.json")))
        except (OSError, ValueError):
            e["closed_loop_units"] = 0
        e["pass"] = e["sft_done"] and e["finite"] and e["audit_nav"] and e["closed_loop_units"] >= 1
        sw[w] = e
    dec = None
    try:
        dec = json.load(open(f"{R}/sweep/decision.json")).get("chosen_w")
    except (OSError, ValueError):
        pass
    v["sweep"] = {"pass": all(e["pass"] for e in sw.values()) and dec is not None, "chosen_w": dec, **sw}
    ab = {}
    for x in ("A", "B"):
        d = f"{S}/v2smoke{x}"
        info = {}
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.startswith("run_info_step"):
                info = json.load(open(os.path.join(d, f)))
        ab[x] = {"done": os.path.exists(f"{d}/DONE"), "finite": finite_log(d), "best": os.path.isdir(f"{d}/best"),
                 "image_features": info.get("image_features"), "state_dim": info.get("state_dim"), "grad_accum": info.get("grad_accum")}
        ab[x]["pass"] = ab[x]["done"] and ab[x]["finite"] and ab[x]["best"]
    ab["B"]["pass"] = ab["B"]["pass"] and any("_h" in k for k in (ab["B"]["image_features"] or [])) and ab["B"]["state_dim"] == 16
    v["ab_train"] = {"pass": ab["A"]["pass"] and ab["B"]["pass"], **ab}
    ev = {}
    for x in ("A", "B"):
        try:
            s = json.load(open(f"{R}/eval_{x}/train_cl/summary.json"))
            ready = "READY" in open(f"{R}/eval_{x}/server.log").read()
            ev[x] = {"eval_count": s["eval_count"], "engineering_errors": s["engineering_errors"], "server_ready": ready,
                     "errors": s.get("engineering_error_units"),
                     "pass": s["eval_count"] >= 1 and s["engineering_errors"] == 0 and ready}
        except (OSError, ValueError, KeyError) as e:
            ev[x] = {"pass": False, "error": repr(e)}
    v["eval"] = {"pass": all(x["pass"] for x in ev.values()), **ev}
    sc = {}
    for x in ("A", "B"):
        try:
            r = json.load(open(f"{R}/consistency_{x}.json"))
            sc[x] = {"pass": bool(r["pass"]), "srv_vs_inproc_fp32": r["diffs"]["srv_fp32 vs in_fp32"]["max_chunk"],
                     "normalizer_vs_train": max(r["saved_normalizer_vs_dataset_max_abs"].values()),
                     "train_path_vs_inproc": r["diffs"]["train_path_fp32 vs in_fp32"]["max_chunk"]}
        except (OSError, ValueError, KeyError) as e:
            sc[x] = {"pass": False, "error": repr(e)}
    v["server_consistency"] = {"pass": all(x["pass"] for x in sc.values()), **sc}
    v["ALL_PASS"] = all(v[k]["pass"] for k in ("gates", "sweep", "ab_train", "server_consistency", "eval"))
    json.dump(v, open(a.out, "w"), indent=1)
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
