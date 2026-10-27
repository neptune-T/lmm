"""Qualify recovered converted data with the real B loader and public model.
Original raw-simulator gates remain historical evidence, not synthetic reruns.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'alphabot2_pi05'))
import ovmm_model as M
import sampling
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from recovered_shard_alignment import frame
from recover_v2_dataset import episodes, write


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    c=sampling.load_config(a.config)
    hist=c['history']
    root=Path(c['train_data'])
    meta=LeRobotDatasetMetadata('local/recovered',root=root)
    cfg=M.build_config(c['init_checkpoint'],history=hist)
    dts=M.history_delta_timestamps(resolve_delta_timestamps(cfg,meta),hist)
    ds=LeRobotDataset('local/recovered',root=root,delta_timestamps=dts,video_backend='pyav')
    ok={}
    assert cfg.chunk_size==50 and M.state_dim(hist)==16
    assert list(meta.features['action']['names'])==list(M.IO.ACTION_LABELS)
    assert list(meta.features['observation.state']['names'])==list(M.IO.STATE_LABELS)
    es=episodes(root)
    info=json.loads((root/'meta/info.json').read_text())
    # Check train-only quantiles without modifying the recovered data/stat files.
    for key,width in [('action',18),('observation.state',16)]:
        arrays=[pq.read_table(f,columns=[key])[key].combine_chunks().values.to_numpy().reshape(-1,width)
                for f in sorted((root/'data').rglob('*.parquet'))]
        values=np.concatenate(arrays).astype(np.float64)
        span=np.maximum(np.ptp(values,axis=0),1e-9)
        for q in [.01,.99]:
            stored=np.asarray(meta.stats[key]['q01' if q==.01 else 'q99'])
            assert np.max(np.abs(stored-np.quantile(values,q,axis=0))/span)<1e-5
        del values,arrays
    ok['exact_train_quantiles']=True
    def batch(index):
        it=ds[index]
        raw={k:(v[None] if torch.is_tensor(v) else [v]) for k,v in it.items()
             if k.startswith(('observation','action','task'))}
        return M.apply_history(raw,hist)
    pixels=0
    for ei in [0,len(es)//2,len(es)-1]:
        e=es[ei]
        for t in [0,min(16,e['length']-1),e['length']-1]:
            b=batch(e['dataset_from_index']+t)
            for off in [0,5,10,15]:
                key='observation.images.camera_top'+(f'_h{off}' if off else '')
                actual=b[key][0].permute(1,2,0).numpy()*255
                wanted=frame(root,info,e,'camera_top',max(0,t-off))
                assert np.max(np.abs(actual-wanted))<1e-3,(ei,t,off)
                pixels+=1
    ok['loader_history_frames']=pixels
    policy,pre,post=M.build(cfg,c['init_checkpoint'],c['init_checkpoint'],meta.stats)
    policy.eval()
    e=es[len(es)//2]
    b=batch(e['dataset_from_index']+e['length']//2)
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        loss,terms=M.configured_loss(policy,pre(dict(b)),b['action_is_pad'],c['loss'])
        pred=post(policy.predict_action_chunk(pre(dict(b))))
    assert torch.isfinite(loss) and pred.shape==(1,50,18) and torch.isfinite(pred).all()
    ok['loss']=float(loss)
    ok['output_shape']=list(pred.shape)
    # Fixed-noise fp32 padding probe: corrupted padded actions cannot alter valid outputs.
    tail=batch(e['dataset_to_index']-10)
    corrupt={k:(v.clone() if torch.is_tensor(v) else v) for k,v in tail.items()}
    pad=tail['action_is_pad'].to('cuda')
    corrupt['action'][tail['action_is_pad']]=100
    x,y=pre(dict(tail)),pre(corrupt)
    torch.manual_seed(77)
    nz=torch.randn(policy.prepare_action(x).shape,device='cuda')
    tm=torch.full((1,),.5,device='cuda')
    with torch.no_grad():
        _,v=M.flow_losses(policy,x,pad,noise=nz,time=tm,return_v=True)
        _,w=M.flow_losses(policy,y,pad,noise=nz,time=tm,return_v=True)
    err=float((v[~pad]-w[~pad]).abs().max())
    assert err<=1e-6,err
    ok['fp32_padding_max_error']=err
    write(a.out,dict(passed=True,checks=ok,config=c,scope='recovered loader/model checks; raw physics not re-run'))
    print('MODEL_PROBE_PASS',json.dumps(ok),flush=True)


if __name__=='__main__':
    main()
