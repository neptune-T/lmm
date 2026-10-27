"""Compare first/middle/last episodes of EVERY recovered conversion shard.

Source here explicitly means the retained converted shard, not lost raw NPZs.
Checks labels, current images and the 0.5/1/1.5-second head-history indices.
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from recover_v2_dataset import episodes, write


def frame(root, info, ep, camera, t):
    key = 'observation.images.' + camera
    rel = info['video_path'].format(video_key=key, chunk_index=ep[f'videos/{key}/chunk_index'], file_index=ep[f'videos/{key}/file_index'])
    stamp = ep[f'videos/{key}/from_timestamp'] + t / info['fps']
    with av.open(str(root / rel)) as c:
        s = c.streams.video[0]
        s.thread_count = 1
        c.seek(int(stamp / s.time_base), stream=s, backward=True)
        for f in c.decode(s):
            if abs(float(f.time) - stamp) < .025:
                return f.to_ndarray(format='rgb24')
            if float(f.time) > stamp + .025:
                break
    raise RuntimeError(f'Frame absent: {root} {camera} {stamp}')


def table(root):
    return pa.concat_tables([pq.read_table(f, columns=['observation.state', 'action']) for f in sorted((root/'data').rglob('*.parquet'))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--staged', required=True)
    ap.add_argument('--shards', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    results = []
    for split, count in [('train', 24), ('val', 2)]:
        dst = Path(a.staged) / ('alphabot2_ovmm_v2_' + split)
        mi = json.loads((dst/'meta/info.json').read_text())
        idx = json.loads((dst/'ovmm_episode_index.json').read_text())
        lookup = {e['unit']: e['episode_index'] for e in idx['episodes']}
        me = episodes(dst)
        mt = table(dst)
        for shard in range(count):
            src = Path(a.shards) / f'{split}_{shard}'
            si = json.loads((src/'meta/info.json').read_text())
            se = episodes(src)
            ix = json.loads((src/'ovmm_episode_index.json').read_text())
            st = table(src)
            for i in sorted({0, len(se)//2, len(se)-1}):
                ent = ix['episodes'][i]
                j = lookup[ent['unit']]
                length = ent['frames']
                scores = []
                checked = 0
                for t in sorted({0, min(16,length-1), length//2, length-1}):
                    left = st.slice(se[i]['dataset_from_index']+t,1).to_pylist()[0]
                    right = mt.slice(me[j]['dataset_from_index']+t,1).to_pylist()[0]
                    for key in left:
                        assert np.allclose(left[key],right[key],rtol=0,atol=1e-6), (split,shard,i,t,key)
                    for camera in ['camera_left','camera_right','camera_top']:
                        times = sorted({t, *[max(0,t-k) for k in [5,10,15]]}) if camera=='camera_top' else [t]
                        for ti in times:
                            x=frame(src,si,se[i],camera,ti).astype(float)
                            y=frame(dst,mi,me[j],camera,ti).astype(float)
                            mse=np.mean((x-y)**2)
                            psnr=99.0 if mse==0 else float(10*np.log10(255**2/mse))
                            assert psnr>=30,(split,shard,i,ti,camera,psnr)
                            scores.append(psnr)
                            checked+=1
                results.append(dict(split=split,shard=shard,unit=ent['unit'],frames_checked=checked,psnr_min=min(scores)))
            print('SHARD_PASS',split,shard,flush=True)
            write(a.out,dict(complete=False,results=results))
    write(a.out,dict(complete=True,passed=True,shards=26,episodes=len(results),
                     source_kind='retained converted shards; not original simulator NPZ',results=results))


if __name__=='__main__':
    main()
