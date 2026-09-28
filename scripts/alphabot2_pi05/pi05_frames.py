import av, numpy as np, pandas as pd
from PIL import Image
D = "/nix/yanjun/zts/alphabot2/hf/dataset"
ep = pd.read_parquet(f"{D}/meta/episodes/chunk-000/file-000.parquet")
cams = ["camera_front", "camera_left", "camera_right", "camera_top"]
picks = [0, 25, 343]
rows = []
for e in picks:
    r = ep[ep.episode_index == e].iloc[0]
    tiles = []
    for c in cams:
        t0 = float(r[f"videos/observation.images.{c}/from_timestamp"]); t1 = float(r[f"videos/observation.images.{c}/to_timestamp"])
        want = [t0 + 0.1, (t0 + t1) / 2]
        cont = av.open(f"{D}/videos/observation.images.{c}/chunk-000/file-000.mp4")
        st = cont.streams.video[0]
        got = []
        for w in want:
            cont.seek(int(max(0, w - 1.0) / st.time_base), stream=st, backward=True, any_frame=False)
            for fr in cont.decode(st):
                if fr.time is not None and fr.time >= w - 0.051:
                    got.append(fr.to_ndarray(format="rgb24")); break
        cont.close()
        tiles.append(np.concatenate(got, axis=0))
    rows.append(np.concatenate(tiles, axis=1))
    print(e, r["tasks"], [t.shape for t in tiles])
Image.fromarray(np.concatenate(rows, axis=0)).save("/nix/yanjun/zts/alphabot2/hf/cam_montage.png")
print("saved")
