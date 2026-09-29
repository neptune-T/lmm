"""Render-quality probe: RTX anti-aliasing/upscaling modes vs image noise and cost.

The base kit runs DLSS in Performance mode (``rtx.post.dlss.execMode = 0``);
at 320x240 that renders ~160x120 internally and upscales, which shows up as
speckles and streaks in every camera. For each candidate setting this renders
fixed scene states (navigation view, manipulation view) and measures:

* speckle fraction -- pixels deviating > 40 (0-255) from their 3x3 median;
* temporal noise   -- mean abs difference between two renders of a static state;
* cost             -- wall seconds per action (20 physics substeps + 4 cameras).
"""
import json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simworld as SW  # noqa: E402

app, ZD = SW.launch()
import carb  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

OUT = os.environ.get("RQ_OUT", "/nix/yanjun/zts/alphabot2/stage1/render_quality")
os.makedirs(OUT, exist_ok=True)
W = SW.SimWorld(app, ZD)
S = carb.settings.get_settings()
AA_VARIANTS = {
    "default_dlss_perf": {},
    "aa_none": {"/rtx/post/aa/op": 0},
    "aa_taa": {"/rtx/post/aa/op": 1},
    "aa_fxaa": {"/rtx/post/aa/op": 2},
    "dlss_quality": {"/rtx/post/aa/op": 3, "/rtx/post/dlss/execMode": 2},
    "dlaa": {"/rtx/post/aa/op": 4},
}
NOGI = {"/rtx/indirectDiffuse/enabled": False, "/rtx/reflections/enabled": False,
        "/rtx/ambientOcclusion/enabled": False, "/rtx/translucency/enabled": False}
#: "_sun_angle", "_extra_renders", "_wrist_near" are handled here, not carb settings
LIGHT_VARIANTS = {
    "base": {},
    "sun_hard": {"_sun_angle": 0.0},
    "no_sampled_lighting": {"/rtx/directLighting/sampledLighting/enabled": False},
    "no_gi": dict(NOGI),
    "no_gi_sun_hard": {**NOGI, "_sun_angle": 0.0},
    "no_gi_sun_hard_nosampled": {**NOGI, "_sun_angle": 0.0, "/rtx/directLighting/sampledLighting/enabled": False},
    "base_extra3": {"_extra_renders": 3},
    "wrist_near_5cm": {"_wrist_near": 0.05},
    "wrist_near_8cm": {"_wrist_near": 0.08},
}
LIGHT2_VARIANTS = {
    "base": {},
    "no_autoexposure": {"/rtx/post/histogram/enabled": False},
    "sun1000_dome": {"_sun_intensity": 1000.0, "_dome": 600.0},
    "pathtrace_spp1_denoise": {"/rtx/rendermode": "PathTracing", "/rtx/pathtracing/spp": 1,
                               "/rtx/pathtracing/totalSpp": 1, "/rtx/pathtracing/optixDenoiser/enabled": True},
    "pathtrace_spp4_denoise": {"/rtx/rendermode": "PathTracing", "/rtx/pathtracing/spp": 4,
                               "/rtx/pathtracing/totalSpp": 4, "/rtx/pathtracing/optixDenoiser/enabled": True},
}
if os.environ.get("VARIANT_SET") == "light2":
    LIGHT_VARIANTS = LIGHT2_VARIANTS
VARIANTS = LIGHT_VARIANTS if os.environ.get("VARIANT_SET", "light") in ("light", "light2") else AA_VARIANTS
KEYS = sorted({k for v in list(AA_VARIANTS.values()) + list(LIGHT_VARIANTS.values()) + list(LIGHT2_VARIANTS.values())
               for k in v if k.startswith("/")})
R = {"initial": {k: S.get(k) for k in KEYS}}
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402
SUN = UsdLux.DistantLight(W.stage.GetPrimAtPath("/World/Sun"))
SUN_ANGLE0 = float(SUN.GetAngleAttr().Get())
SUN_I0 = float(SUN.GetIntensityAttr().Get())
DOME = UsdLux.DomeLight.Define(W.stage, "/World/RQDome")
DOME.CreateIntensityAttr(0.0)
EXTRA = 0


def set_wrist_near(near):
    for k in ("cam_left_wrist", "cam_right_wrist"):
        UsdGeom.Camera(W.B.cameras[k].prim).GetClippingRangeAttr().Set(Gf.Vec2f(near, 100.0))


def observe_imgs():
    for _ in range(EXTRA):
        W.world.render()
    return {k: W.B.render(k) for k in W.cams}


def scene():
    W.place_table("table_a", 1.5, 0.0, 0.0)
    W.place_table("table_b", 0.0, 1.9, math.pi / 2)
    W.place_box(1.40, -0.20, 0.0)
    W.place_object("red cube", 1.40, 0.13, 0.3)


def speckle(img):
    g = img.astype(float).mean(axis=2)
    pad = np.pad(g, 1, mode="edge")
    stack = np.stack([pad[i:i + g.shape[0], j:j + g.shape[1]] for i in range(3) for j in range(3)])
    return float(np.mean(np.abs(g - np.median(stack, axis=0)) > 40))


rows, montage = [], []
for name, sets in VARIANTS.items():
    for k, v in R["initial"].items():            # back to the initial values first
        if v is not None:
            S.set(k, v)
    SUN.GetAngleAttr().Set(sets.get("_sun_angle", SUN_ANGLE0))
    SUN.GetIntensityAttr().Set(sets.get("_sun_intensity", SUN_I0))
    DOME.GetIntensityAttr().Set(sets.get("_dome", 0.0))
    EXTRA = int(sets.get("_extra_renders", 0))
    set_wrist_near(sets.get("_wrist_near", 0.02))
    for k, v in sets.items():
        if k.startswith("/"):
            S.set(k, v)
    row = {"variant": name, "settings": sets}
    ims = []
    for label, base in (("nav", (0.0, 0.0)), ("manip", (0.95, -0.12))):
        W.reset_robot(base, 0.0, scene=scene)
        for _ in range(4):
            W.apply_joint_action(W.A.home_action())
        a = observe_imgs()
        W.apply_joint_action(W.A.home_action())
        b = observe_imgs()
        row[f"{label}_black_frac"] = {k: round(float(np.mean(a[k].max(axis=2) < 8)), 3) for k in W.cams}
        row[f"{label}_speckle"] = {k: round(speckle(a[k]), 4) for k in W.cams}
        row[f"{label}_temporal_noise"] = {k: round(float(np.abs(a[k].astype(float) - b[k]).mean()), 2) for k in W.cams}
        ims.append(np.concatenate([a["cam_front"], a["cam_left_wrist"], a["cam_right_wrist"], a["cam_chest"]], axis=1))
    t0 = time.time()
    for _ in range(20):
        W.apply_joint_action(W.A.home_action())
        observe_imgs()
    row["wall_s_per_action"] = round((time.time() - t0) / 20, 4)
    row["speckle_mean"] = round(float(np.mean([v for l in ("nav", "manip") for v in row[f"{l}_speckle"].values()])), 4)
    row["temporal_noise_mean"] = round(float(np.mean([v for l in ("nav", "manip") for v in row[f"{l}_temporal_noise"].values()])), 2)
    rows.append(row)
    montage.append(np.concatenate(ims, axis=0))
    print(json.dumps(row), flush=True)
    Image.fromarray(montage[-1]).save(f"{OUT}/{name}.png")
R["variants"] = rows
print("###RQ###" + json.dumps(R), flush=True)
json.dump(R, open(f"{OUT}/render_quality.json", "w"), indent=1)
app.close()
