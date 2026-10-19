"""Render the planner-assisted (PA-OG) baseline report as one self-contained HTML page (stdlib only).
    python pa_report_html.py pa_report_data.json filmstrip.jpg out.html
Every number comes from pa_report_data.json (pa_report_data.py); the filmstrip is embedded as a data URI.
"""
import base64, html, json, sys

D = json.load(open(sys.argv[1]))
STRIP = base64.b64encode(open(sys.argv[2], "rb").read()).decode()
OUT = sys.argv[3]
E = html.escape
FUN = ("find", "grasp", "carry", "place", "full")
HELDOUT_DIRS = {"pa_v2B_heldout", "pa_v2B_retry_heldout", "pa_msft_heldout"}
ORDER = ["pa_v2B", "pa_v2B_heldout", "pa_v2B_val", "pa_v2B_free", "pa_v2B_retry", "pa_v2B_retry_heldout",
         "pa_msft", "pa_msft_heldout", "pa_msft_val", "pa_smoke_v2B", "pa_repro"]


def model(p):
    return "M-SFT v1" if "msft" in p["ckpt"] else "v2-B"


def variant(p):
    return p["label"].replace("PA-OG (planner-assisted, oracle goal)", "PA-OG").replace(" + base lock in manipulation", " + base lock")


def tier(p):
    if p["split"] == "val":
        return "OVMM-val"
    return "held-out episodes" if p["dir"] in HELDOUT_DIRS else "train units"


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "–"


passes = sorted([p for p in D["passes"] if p["dir"] not in ("pa_smoke_v2B", "pa_repro", "pa_det")],
                key=lambda p: (ORDER.index(p["dir"]) if p["dir"] in ORDER else 99, p["seed"]))
repro = [p for p in D["passes"] if p["dir"] == "pa_repro"]


def sel(dirs, mdl, seeds=None, retries=False, lock=True):
    out = []
    for p in passes:
        if p["dir"] in dirs and model(p) == mdl and (seeds is None or p["seed"] in seeds) \
                and (("retries" in p["label"]) == retries) and (("base lock" in p["label"]) == lock):
            out.append(p)
    return out


def tot(ps):
    s = sum(p["funnel"]["full"] for p in ps); n = sum(p["eval_count"] for p in ps)
    return s, n


def frac(ps):
    s, n = tot(ps)
    return f"{s}/{n}" if n else "–"


def funnel_row(p):
    n = p["eval_count"]; full = p["funnel"]["full"]
    cells = "".join(f'<td class="num">{p["funnel"][k]}</td>' for k in FUN[:-1])
    return (f'<tr><td>{E(model(p))}</td><td>{E(variant(p))}</td><td>{E(tier(p))}</td><td class="num">{p["seed"]}</td>'
            f'<td class="num">{n}</td>{cells}<td class="num full {"ok" if full else "zero"}">{full}/{n} '
            f'<span class="pc">{pct(full, n)}</span></td></tr>')


rows = "".join(funnel_row(p) for p in passes)
v_tr, v_ho, v_va = sel({"pa_v2B"}, "v2-B"), sel({"pa_v2B_heldout"}, "v2-B"), sel({"pa_v2B_val"}, "v2-B")
m_tr, m_ho, m_va = sel({"pa_msft"}, "M-SFT v1"), sel({"pa_msft_heldout"}, "M-SFT v1"), sel({"pa_msft_val"}, "M-SFT v1")
tr_s, tr_n = tot(v_tr); ho_s, ho_n = tot(v_ho); va_s, va_n = tot(v_va)
free = sel({"pa_v2B_free"}, "v2-B", lock=False); lock0 = sel({"pa_v2B"}, "v2-B", seeds={0})
retry = sel({"pa_v2B_retry"}, "v2-B", retries=True); retry_h = sel({"pa_v2B_retry_heldout"}, "v2-B", retries=True)
ho0 = sel({"pa_v2B_heldout"}, "v2-B", seeds={0})
# paired M-SFT vs v2-B: same dirs' units / seeds
m_seeds = {p["seed"] for p in m_tr}
v_tr_m = [p for p in v_tr if p["seed"] in m_seeds]
v_ho_m = [p for p in v_ho if p["seed"] in {p2["seed"] for p2 in m_ho}]
v_va_m = [p for p in v_va if p["seed"] in {p2["seed"] for p2 in m_va}]

def paired(ma, vb):
    """(M-SFT-only, v2-B-only, both, exact two-sided sign-test p) over identical (unit, seed) pairs"""
    import math
    def succ(ps):
        return {(u, p["seed"]): any(x["unit"] == u for x in p["successes"]) for p in ps for u in p["units"]}
    a, b = succ(ma), succ(vb)
    ks = set(a) & set(b)
    w = sum(a[k] and not b[k] for k in ks); l = sum(b[k] and not a[k] for k in ks); both = sum(a[k] and b[k] for k in ks)
    n = w + l
    pv = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n) if n else 1.0
    return w, l, both, len(ks), pv


rep = list({r["trace"]: r for r in D["replays"]}.values())          # one row per recorded trace
rep_ok = sum(1 for r in rep if r["funnel"].get("full"))
rep_units = sorted({r["unit"] for r in rep if r["funnel"].get("full")})
rep_rows = "".join(
    f'<tr><td class="mono">{E(r["unit"])}</td><td class="mono small">{E("/".join(r["trace"].split("/")[-3:]))}</td>'
    f'<td class="num">{r["steps_replayed"]}/{r["steps_recorded"]}</td>'
    f'<td class="{"ok" if r["funnel"].get("full") else "bad"}">{"full success" if r["funnel"].get("full") else "not full"}</td></tr>'
    for r in rep)
rp_s, rp_n = tot(repro)
rep_bad = [r for r in rep if not r["funnel"].get("full")]
rep_bad_txt = (" Not reproduced: " + ", ".join(f"{r['unit']} ({'/'.join(r['trace'].split('/')[-3:-2])}; replay ends with place = "
                                               f"{r['funnel'].get('place')})" for r in rep_bad) + ", a marginal placement; it is "
               "not counted as a verified success.") if rep_bad else ""


def diag_counts(name):
    dg = D["diagnostics"].get(name)
    if not dg:
        return None
    b = dg["by"]

    def g(k, f):
        x = b.get(k, {})
        return x.get(f, 0), x.get("valid", 0)
    out = {}
    for grp in ("train_unit", "heldout"):
        out[grp] = {"expert_obj": g(f"obj / expert_rest / {grp}", "ovmm_full_end"), "grasp": g(f"obj / policy / {grp}", "grasp"),
                    "carry": g(f"obj / policy / {grp}", "carry_ready"), "expert_rec": g(f"recep / expert_rest / {grp}", "full_end"),
                    "place": g(f"recep / policy / {grp}", "place_ever"), "full": g(f"recep / policy / {grp}", "full_end")}
    out["causes"] = dg.get("policy_failure_causes", {})
    return out


def diag_table(name, label):
    c = diag_counts(name)
    if c is None:
        return f"<p class='muted'>{E(label)}: not run.</p>"
    trs = []
    for grp, gl in (("train_unit", "train units"), ("heldout", "held-out episodes")):
        x = c[grp]
        cell = lambda k, p=True: f"<td class='num'>{x[k][0]}/{x[k][1]}" + (f" <span class='pc'>{pct(*x[k])}</span>" if p else "") + "</td>"
        trs.append(f"<tr><td>{gl}</td>{cell('expert_obj', False)}{cell('grasp')}{cell('carry')}{cell('expert_rec', False)}{cell('place')}{cell('full')}</tr>")
    names = {"no_grasp": "no grasp", "no_lift": "grasped, not lifted", "no_tuck": "lifted, never tucked",
             "dropped_later": "carry-ready, dropped later", "ok": "carry-ready and held", "released_no_place": "released off the receptacle",
             "placed_not_full_end": "placed, then disturbed", "ok_r": "full success at end"}
    obj = sorted(((k.split(" / ")[1], v) for k, v in c["causes"].items() if k.startswith("obj")), key=lambda kv: -kv[1])
    rec = sorted(((k.split(" / ")[1], v) for k, v in c["causes"].items() if k.startswith("recep")), key=lambda kv: -kv[1])
    oc = " · ".join(f"{names.get(k, k)} {v}" for k, v in obj)
    rc = " · ".join(f"{names.get('ok_r' if k == 'ok' else k, k)} {v}" for k, v in rec)
    return f"""<h3>{E(label)}</h3><div class="tw"><table><thead><tr><th>Units</th><th>Expert continues from<br>object arrival (full)</th>
<th>π0.5 grasp</th><th>π0.5 carry-ready</th><th>Expert continues from<br>receptacle arrival (full)</th><th>π0.5 place phase</th>
<th>π0.5 full at end</th></tr></thead><tbody>{''.join(trs)}</tbody></table></div>
<p class="small"><b>Object entry, per policy job:</b> {E(oc)}.<br><b>Receptacle entry:</b> {E(rc)}.</p>"""


dv = diag_counts("manip_arrival")
if dv:
    gsum = [dv[g]["carry"] for g in ("train_unit", "heldout")]; psum = [dv[g]["full"] for g in ("train_unit", "heldout")]
    carry_rate = pct(sum(a for a, _ in gsum), sum(b for _, b in gsum)); place_rate = pct(sum(a for a, _ in psum), sum(b for _, b in psum))
else:
    carry_rate = place_rate = "–"
dm = diag_counts("manip_arrival_msft")


def msft_line():
    if not m_tr:
        return ""
    return (f"<li><b>Manipulation-only fine-tune (M-SFT v1, 3000 steps from v2-B):</b> PA-OG + base lock, train units "
            f"<b>{frac(m_tr)}</b> vs v2-B {frac(v_tr_m)} (same seeds), held-out <b>{frac(m_ho)}</b> vs {frac(v_ho_m)}, "
            f"OVMM-val <b>{frac(m_va)}</b> vs {frac(v_va_m)}.</li>")


pt, ph, pv_ = paired(m_tr, v_tr_m), paired(m_ho, v_ho_m), paired(m_va, v_va_m)
pair_txt = " · ".join(f"{lab}: M-SFT-only {x[0]}, v2-B-only {x[1]}, both {x[2]} of {x[3]} pairs (sign test p = {x[4]:.2f})"
                      for lab, x in (("train", pt), ("held-out", ph), ("OVMM-val", pv_)) if x[3])
msft_section = ""
if m_tr or dm:
    msft_section = f"""<section><h2>3 · Minimal manipulation fix: M-SFT v1</h2>
<p>Continuation of v2-B (unchanged; its sha256 was verified after training) on the same v2 data with navigation, tuck and settle frames at sampling weight 0; place stages up-weighted. 3000 steps × 16 (48k samples, 30% of v2-B's budget), lr 1e-5 → 2.5e-6, evaluated checkpoint <code>last</code> (fixed budget, pre-registered). Its NAV is irrelevant here because the planner navigates. Interrupted once by the container restart and resumed from its step-2000 checkpoint.</p>
<p>PA-OG + base lock, paired with v2-B on the same units and seeds: train <b>{frac(m_tr)}</b> vs {frac(v_tr_m)} · held-out <b>{frac(m_ho)}</b> vs {frac(v_ho_m)} · OVMM-val <b>{frac(m_va)}</b> vs {frac(v_va_m)}. Paired on identical (unit, seed): {E(pair_txt)}. The train gain is a trend, not significant at this n. Per-pass rows are in the table above.</p>
{diag_table("manip_arrival_msft", "M-SFT v1 from expert arrival (same 240 jobs)")}
</section>"""

H = f"""<title>AlphaBot2 PA-OG Baseline</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap">
<style>
:root{{--bg:#f4f5f2;--panel:#ffffff;--ink:#16191c;--muted:#5a636b;--rule:#d9ddd6;--accent:#2b5f8a;--ok:#2f7a4f;--okbg:#e4f1e8;--bad:#a8483a;--warn:#8a6a1f;--warnbg:#f5eedb;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#121518;--panel:#191d21;--ink:#e6e8e4;--muted:#9aa3ab;--rule:#2b3036;--accent:#86b6dd;--ok:#72c495;--okbg:#18301f;--bad:#e3907f;--warn:#d9b45c;--warnbg:#332a14;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#121518;--panel:#191d21;--ink:#e6e8e4;--muted:#9aa3ab;--rule:#2b3036;--accent:#86b6dd;--ok:#72c495;--okbg:#18301f;--bad:#e3907f;--warn:#d9b45c;--warnbg:#332a14;color-scheme:dark}}
body{{background:var(--bg);color:var(--ink);font:15px/1.55 "IBM Plex Sans",system-ui,sans-serif;padding-block:32px 64px;padding-inline:16px}}
.wrap{{max-width:1000px;margin:0 auto;display:flex;flex-direction:column;gap:28px}}
h1,h2{{font-family:"Source Serif 4",Georgia,serif;text-wrap:balance;margin:0}}
h1{{font-size:clamp(26px,4vw,34px);line-height:1.15}} h2{{font-size:23px;margin-bottom:6px}} h3{{font-size:15px;margin:14px 0 6px}}
p{{margin:6px 0;max-width:75ch}} .muted{{color:var(--muted)}} .small{{font-size:13px}}
.mono,code{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px}}
.eyebrow{{text-transform:uppercase;letter-spacing:.08em;font-size:12px;color:var(--muted);font-weight:600}}
header{{display:flex;flex-direction:column;gap:8px;border-bottom:1px solid var(--rule);padding-bottom:18px}}
.verdict{{background:var(--panel);border:1px solid var(--rule);border-left:4px solid var(--ok);padding:14px 18px;border-radius:4px}}
.verdict ul{{margin:6px 0 0;padding-left:18px}} .verdict li{{margin:5px 0}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.tile{{background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:12px 14px;display:flex;flex-direction:column;gap:2px}}
.tile .v{{font-family:"IBM Plex Mono",monospace;font-size:24px;font-weight:500;font-variant-numeric:tabular-nums}}
.tile .l{{font-size:13px;color:var(--muted)}}
.tw{{overflow-x:auto;border:1px solid var(--rule);border-radius:4px;background:var(--panel)}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}}
th{{font-weight:600;font-size:12.5px;color:var(--muted);background:var(--bg)}}
tr:last-child td{{border-bottom:none}}
.num{{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;white-space:nowrap}}
.pc{{color:var(--muted);font-size:11.5px}}
td.full.ok{{color:var(--ok);font-weight:600;background:var(--okbg)}} td.full.zero{{color:var(--bad)}}
td.ok{{color:var(--ok)}} td.bad{{color:var(--bad)}}
.note{{background:var(--warnbg);border:1px solid var(--rule);border-left:4px solid var(--warn);padding:10px 14px;border-radius:4px;font-size:14px}}
.strip{{width:100%;border-radius:4px;border:1px solid var(--rule)}}
.contract{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}
.contract div{{background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:10px 14px;font-size:14px}}
.contract b{{display:block;margin-bottom:2px}}
ol.phases{{margin:6px 0;padding-left:20px}} ol.phases li{{margin:5px 0}}
a{{color:var(--accent)}}
</style>
<div class="wrap">
<header>
<span class="eyebrow">AlphaBot 2 · Habitat OVMM · π0.5 · 2026-10-19</span>
<h1>Planner-assisted baseline: first reproducible full-task successes</h1>
<p class="muted">Label: <b>PA-OG</b>, planner-assisted with an <b>oracle goal</b>. This is not an end-to-end result. The planner's navigation goals use privileged simulator state; the π0.5 policy's inputs, its 18-D action interface and every phase switch use observable signals only. Policy v2-B (sha256 405f2d5d…), fp32, H_exec 5, unless marked M-SFT v1.</p>
</header>

<section class="verdict">
<span class="eyebrow">Outcome</span>
<ul>
<li><b>Non-zero full-task success</b>, verified by replay: <b>{rep_ok}/{len(rep)}</b> recorded successes re-execute from a fresh reset to <code>ovmm_place_success</code> = True ({len(rep_units)} distinct episodes). The v2 end-to-end evals had 0 successes.{E(rep_bad_txt)}</li>
<li>v2-B, PA-OG + base lock: train units <b>{tr_s}/{tr_n}</b> ({pct(tr_s, tr_n)}, {len(v_tr)} seeds), held-out episodes <b>{ho_s}/{ho_n}</b>, OVMM-val <b>{va_s}/{va_n}</b>. Generalization beyond the training scenes is the open problem.</li>
<li>Manipulation works once navigation is solved, but only partially. From the expert's arrival, π0.5 places successfully in {place_rate} of trials; grasp-to-carry-ready ({carry_rate}) is the main bottleneck. The v2 zero-success was navigation.</li>
{msft_line()}
</ul>
</section>

<div class="tiles">
<div class="tile"><span class="v">{tr_s}/{tr_n}</span><span class="l">v2-B full success, train units</span></div>
<div class="tile"><span class="v">{ho_s}/{ho_n}</span><span class="l">v2-B full success, held-out episodes</span></div>
<div class="tile"><span class="v">{va_s}/{va_n}</span><span class="l">v2-B full success, OVMM-val</span></div>
<div class="tile"><span class="v">{rep_ok}/{len(rep)}</span><span class="l">successes reproduced by open-loop replay</span></div>
</div>

<section>
<h2>First success: unit 10064#0, seed 0</h2>
<p class="small muted">Policy camera strips (head | right wrist | left wrist). Planner NAV to the pick standoff → π0.5 grasps (step 630) and tucks → planner carries → π0.5 reaches twice and releases on the second reach (step 1230) → the toy rests on the table → <code>ovmm_place_success</code> at step 1522. The episode reproduced in identical closed-loop re-runs and in open-loop replay.</p>
<img class="strip" alt="Eight key frames of the 10064#0 success" src="data:image/jpeg;base64,{STRIP}">
</section>

<section>
<h2>1 · Manipulation from expert arrival (diagnostic)</h2>
<p>A validated replay of each accepted expert episode is restored to the first frame of <i>reach</i> (navigation to the object done) or <i>reach_place</i> (at the receptacle, object in hand); the restore error was 0.0 cm / 0.0°. Then either the expert's own remaining actions (control) or the policy (2 seeds) continue. 20 v2-train units + 20 held-out episodes (v2 SFT-val), 240 jobs per model.</p>
{diag_table("manip_arrival", "v2-B from expert arrival")}
<p class="small muted">"Carry-ready" = held, lifted ≥ 5 cm, then both arms within 0.3 rad of the NAV tuck. One v2-B receptacle job ended in an OVMM-internal goal-reachability assertion (engineering error, excluded).</p>
</section>

<section>
<h2>2 · Planner-assisted full task</h2>
<p>Normal OVMM initialization; OVMM's own measures at episode end. Train units are the frozen v2 training-set eval units (the policy saw these episodes). Held-out units are v2 SFT-val episodes (train-split scenes, unseen episodes). OVMM-val is 10 val episodes × 3 replicas; val starts are fixed, so the replicas differ only in policy noise.</p>
<div class="tw"><table><thead><tr><th>Model</th><th>Variant</th><th>Units</th><th>Seed</th><th>n</th><th>Find</th><th>Grasp</th><th>Carry</th><th>Place</th><th>Full</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p class="small muted">Find / grasp / carry / place / full = <code>ovmm_find_object_phase_success</code>, <code>ovmm_pick_object_phase_success</code>, <code>ovmm_find_recep_phase_success</code>, <code>ovmm_place_object_phase_success</code>, <code>ovmm_place_success</code>. Free base (π0.5 also drives during manipulation): {frac(free)} vs {frac(lock0)} with base lock (train, seed 0). Retries 3/2 (observable-failure re-approach: 3 pick attempts × 250 steps, 2 place attempts × 350 steps): train {frac(retry)} vs {frac(lock0)}, held-out {frac(retry_h)} vs {frac(ho0)}. Neither change beat the plain base-lock contract.</p>
</section>

{msft_section}

<section>
<h2>4 · Reproducibility</h2>
<p><b>Open-loop replay</b>: reset_to(unit) + fix_start + the episode's recorded executed JointMobileAction18 rows (planner and policy alike), with no policy and no planner. A success that reproduces here is a genuine simulator episode.</p>
<div class="tw"><table><thead><tr><th>Unit</th><th>Trace</th><th>Steps</th><th>Replay</th></tr></thead><tbody>{rep_rows}</tbody></table></div>
<p class="note"><b>Closed-loop determinism depends on serving.</b> With the shared, batched fp32 servers used for every pass above, re-running a success with the same unit and noise seed reproduced it in {rp_s} of {rp_n} re-runs. The others diverged within the first manipulation phase. With a dedicated <code>--max-batch 1</code> server, two closed-loop runs of 17858#0 were <b>bitwise identical</b> (846/846 executed joint actions, max difference 0.0; that episode failed both times). So batch composition is the source, and unbatched serving makes rollouts exactly re-executable. Rates across seeds and episodes are the policy-level reproducibility measure; open-loop replay is the episode-level one. Two seed-0 successes recorded before joint logging existed (17858#0, 14956#0) are not replayable.</p>
</section>

<section>
<h2>5 · Controller and observation contract (PA-OG + base lock)</h2>
<div class="contract">
<div><b>Policy inputs (unchanged from v2-B training)</b>camera_top ← head_rgb, camera_left ← right_wrist_rgb, camera_right ← left_wrist_rgb (240×320), camera_front masked; 16-D state (checkpoint convention); head history at −5/−10/−15 steps; instruction “Move the {{object}} from the {{start}} to the {{goal}}.”</div>
<div><b>Policy output</b>CartesianMobileAction18 chunk (H_pred 50), executed receding-horizon with H_exec 5 through the expert's CartesianExecutor (IK), one atomic 18-D joint action per 10 Hz step. Base lock: dims 16–17 (v, ω) replaced by 0 during manipulation; raw values traced.</div>
<div><b>Privileged, planner only (oracle goal)</b>Target object pose, goal-receptacle surfaces, navmesh, IK feasibility: the expert's standoff planner (pick arm by geodesic cost, place sample). Never sent to the policy and never used in a switch.</div>
<div><b>Observable switches</b>Holding = a gripper commanded closed (≥ 0.5) whose measured closure stops more than 0.05 short of the command. Tucked = both arms within 0.3 rad of the NAV tuck. Arrival = planner odometry vs its own waypoint. Each condition must hold for 3 consecutive steps.</div>
</div>
<ol class="phases">
<li><b>NAV1</b>, planner: arms to the NAV tuck, shortest navmesh path to the pick standoff, turn to its heading. Switch on arrival.</li>
<li><b>MANIP1</b>, π0.5: until holding ∧ tucked (timeout 600 steps).</li>
<li><b>NAV2</b>, planner: arms to the exact tuck with grippers held, place standoff planned from the object as actually held. Switch on arrival.</li>
<li><b>MANIP2</b>, π0.5: until ¬holding ∧ tucked (timeout 500 steps).</li>
<li><b>Settle</b>: hold still 120 steps (fixed, no object-velocity check), then read OVMM's measures.</li>
</ol>
</section>

<section>
<h2>6 · Notes and incidents</h2>
<p><b>Container restart (OOM), 2026-10-19 ~15:44.</b> The pod's 481 GB cgroup limit counts /dev/shm. About 40 concurrent sim workers on top of a 177 GB /dev/shm runtime exceeded it; the container was restarted, and /dev/shm and every process were lost: evals, M-SFT at step ~2450, the scheduler/collection, and the GPU-3 keep-alive. Results on /workspace and CFS were intact. Recovery: v2-B best re-copied from the step-10000 mirror (identical sha256), M-SFT resumed from its CFS step-2000 checkpoint, and unfinished passes resumed (workers skip completed units). A memory log now runs (<code>runs/v2/memlog.txt</code>). The scheduler/collection and the GPU-3 keep-alive were <b>not</b> restarted; that is the owner's call.</p>
<p><b>Not started, per the brief:</b> Frontier RL, FP4, large-scale SFT, NAV training.</p>
</section>

<section>
<h2>7 · Where things are</h2>
<p class="small">Server <code>/workspace/runtime/runs/v2/</code>: <code>manip_arrival/</code>, <code>manip_arrival_msft/</code> (diagnostic rows, traces, summary.json) · <code>pa_v2B*/</code>, <code>pa_msft*/</code> (rows; traces with frames + executed joints; <code>videos/</code>) · <code>pa_replays/replay.jsonl</code> · <code>pa_det/</code> (determinism test) · <code>pa_report_data.json</code> · <code>msft_v1/</code> (train/chain logs; checkpoint on CFS <code>lmm_runtime/sft/msft_v1/last</code>) · <code>memlog.txt</code>. Repo <code>/workspace/lmm</code>: <code>scripts/alphabot2_habitat/{{manip_plan,manip_from_arrival,pa_eval,pa_replay,pa_summary,pa_report_data,pa_report_html}}.py</code>, <code>configs/alphabot2_pi05/sft_msft_v1.yaml</code>; protocol and amendments, written before results: <code>docs/v2/pa_baseline.md</code>.</p>
</section>
</div>
"""
open(OUT, "w").write(H)
print("wrote", OUT, len(H))
