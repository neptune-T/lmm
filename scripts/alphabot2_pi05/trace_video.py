"""Render a pa_eval / manip_from_arrival trace (.npz with frames + frame_steps) as an mp4: the half-res
[camera_top | camera_left | camera_right] strips the policy saw, with step, controller phase, privileged held flag
(logged only) and the executed gripper / base commands burnt in.
    python trace_video.py --trace T.npz --out OUT.mp4 [--title TEXT] [--fps 10]
"""
import argparse, os

import av
import numpy as np
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()
    d = np.load(a.trace)
    fr, fs = d["frames"], d["frame_steps"]
    act, held = d["act"], d["held"]
    phase = [str(p) for p in d["phase"]] if "phase" in d.files else None
    c = av.open(a.out, "w")
    s = c.add_stream("libx264", rate=a.fps)
    s.width, s.height, s.pix_fmt = 960, 240 + 40, "yuv420p"
    s.options = {"crf": "23"}
    for i in range(len(fr)):
        t = min(int(fs[i]), len(act) - 1)
        im = Image.new("RGB", (960, 280))
        im.paste(Image.fromarray(fr[i]).resize((960, 240)), (0, 40))
        dr = ImageDraw.Draw(im)
        dr.text((4, 2), a.title[:150], fill=(255, 255, 255))
        x = act[t]
        txt = f"t={t:4d} phase={phase[t] if phase else '-':8s} held(priv)={int(held[t])}"
        if np.isfinite(x).all():
            txt += f"  gripL={x[14]:.2f} gripR={x[15]:.2f} v={x[16]:+.2f} w={x[17]:+.2f}"
        dr.text((4, 16), txt, fill=(255, 255, 0))
        dr.text((4, 28), "head (camera_top) | right wrist (camera_left) | left wrist (camera_right)", fill=(150, 200, 255))
        for p in s.encode(av.VideoFrame.from_image(im)):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()
    print("VIDEO", a.out, len(fr), "frames", os.path.getsize(a.out), "bytes")


if __name__ == "__main__":
    main()
