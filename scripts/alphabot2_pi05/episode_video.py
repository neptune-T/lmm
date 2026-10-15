"""Render one collected (accepted) episode as an mp4 for inspection: the three recorded cameras side by side
(head_rgb | left_wrist_rgb | right_wrist_rgb, 240x320 each, 10 fps) with the step, expert stage, base v / omega,
gripper commands, arm and instruction burnt in. Reads the raw episode dir (images.npz, lowdim.npz,
metadata.npz, meta.json) exactly as the converter does.

    python episode_video.py --dir EPISODE_DIR --out OUT.mp4
"""
import argparse, json, os

import av
import numpy as np
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ims = np.load(os.path.join(a.dir, "images.npz"))
    act = np.load(os.path.join(a.dir, "lowdim.npz"))["action"]
    stg = [str(s) for s in np.load(os.path.join(a.dir, "metadata.npz"))["stage"]]
    meta = json.load(open(os.path.join(a.dir, "meta.json")))
    cams = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
    T = len(act)
    c = av.open(a.out, "w")
    s = c.add_stream("libx264", rate=10)
    s.width, s.height, s.pix_fmt = 960, 240 + 44, "yuv420p"
    s.options = {"crf": "23"}
    for t in range(T):
        im = Image.new("RGB", (960, 284))
        im.paste(Image.fromarray(np.concatenate([ims[k][t] for k in cams], 1)), (0, 44))
        d = ImageDraw.Draw(im)
        d.text((4, 2), f"{meta.get('instruction', '')}   arm={meta.get('arm')}   unit={meta.get('episode_id')}", fill=(255, 255, 255))
        d.text((4, 16), f"t={t:4d}/{T}  stage={stg[t]:24s} v={act[t, 16]:+.2f}  w={act[t, 17]:+.2f}  "
                        f"gripL={act[t, 14]:.1f} gripR={act[t, 15]:.1f}", fill=(255, 255, 0))
        d.text((4, 30), "head (camera_top)                          left wrist (-> camera_right)            right wrist (-> camera_left)",
               fill=(150, 200, 255))
        for p in s.encode(av.VideoFrame.from_image(im)):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()
    print("VIDEO", a.out, T, "frames", os.path.getsize(a.out), "bytes")


if __name__ == "__main__":
    main()
