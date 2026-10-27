"""Gated dual-A800 reconstruction of v2-B; never launches RL or collection.
Launch inside server tmux. Current explicit user authorization: 10k SFT on
qualified recovered data. The historical v2-A/B weights are not being resumed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True)
    a=ap.parse_args()
    run=Path(a.run)
    lock=open(run/'train_chain.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def event(name,**kw):
        row=dict(time=time.strftime('%FT%T%z'),event=name,**kw)
        with open(run/'events.jsonl','a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1','OMP_NUM_THREADS':'4',
         'OPENBLAS_NUM_THREADS':'2','TORCHDYNAMO_DISABLE':'1','NCCL_IB_DISABLE':'1',
         'TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_PROGRESS_BARS':'1',
         'PI05_SPM':str(run/'assets/paligemma_tokenizer.model'),
         'HF_SYNC_STAGE':str(run/'backups/hf_stage'),'HF_UPLOAD_WORKERS':'2',
         'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
    code=Path(__file__).resolve().parent
    def command(cmd,log):
        with open(run/'logs'/log,'a') as f:
            p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(f'{log} exited {p.returncode}')
    try:
        event('WAIT_GATES')
        required={'checks/data/recovered_integrity.json':'pass','checks/shard_alignment.json':'passed',
                  'checks/model_probe.json':'passed'}
        deadline=time.monotonic()+24*3600
        while True:
            ready=True
            for filename,key in required.items():
                p=run/filename
                if not p.exists():ready=False;continue
                d=json.loads(p.read_text())
                if filename.endswith('shard_alignment.json') and not d.get('complete'):
                    ready=False;continue
                if not d.get(key):raise RuntimeError('failed gate: '+filename)
            backup=run/'checks/data_hf_verified.json'
            if not backup.exists():ready=False
            else:
                d=json.loads(backup.read_text())
                assert not d['mismatch'] and d['verified']==d['files']
            tests=(run/'logs/resume_tests.log').read_text() if (run/'logs/resume_tests.log').exists() else ''
            if 'TEST_EXIT=0' not in tests:ready=False
            if 'TEST_EXIT=' in tests and 'TEST_EXIT=0' not in tests:raise RuntimeError('resume unit tests failed')
            if ready:break
            if time.monotonic()>deadline:raise RuntimeError('gate wait exceeded 24 hours')
            time.sleep(15)
        assets=json.loads((run/'assets/ASSETS.json').read_text())
        assert assets['revision']=='cf43edfb58116c241b298a26342f800bf436d1d6'
        assert not any(v.startswith('FAIL') for v in assets['results'].values())
        # GPU0's explicitly owned guard is permitted during initial model loading;
        # it yields when CUDA sees the training process and resumes if training exits.
        guard=json.loads(Path('/root/lmm-runtime/gpu0_status.json').read_text())
        owned=set(guard.get('host_gpu_pids',[]))
        for gpu in [0,1]:
            ids=set(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split())
            assert not (ids-owned if gpu==0 else ids),f'foreign GPU{gpu} occupancy'
        assert __import__('shutil').disk_usage(run).free > 200*(1<<30),'insufficient CFS headroom'
        event('GATES_PASS',world=2,global_batch=16,steps=10000)
        cfg=run/'sft_v2B_ddp.yaml'
        train=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
               str(code/'alphabot2_pi05/sft_ovmm.py'),'--config',str(cfg)]
        out=run/'sft/v2B'
        ck=out/'ckpt_step0000002'
        if not (ck/'COMPLETE').exists():
            event('DDP_CHECKPOINT_PROBE_START')
            command(train+['--stop-after-step','2'],'sft_probe.log')
        assert (ck/'COMPLETE').exists()
        event('CHECKPOINT_2_COMPLETE')
        command([sys.executable,str(code/'recovered_checkpoint_backup.py'),'--run',str(run),'--checkpoint',str(ck)],'checkpoint_backup.log')
        event('CHECKPOINT_2_BACKED_UP')
        # A named tmux window owns the serialized checkpoint uploader.
        import shlex
        watch=' '.join(shlex.quote(x) for x in [sys.executable,'-u',str(code/'recovered_checkpoint_backup.py'),'--run',str(run),'--watch'])
        watch+=' >> '+shlex.quote(str(run/'logs/checkpoint_backup.log'))+' 2>&1'
        subprocess.run(['tmux','new-window','-t','lmm-recover','-n','checkpoint-backup',watch],check=True,env=env)
        if not (out/'DONE').exists():
            event('SFT_RESUME_TO_10K_START')
            command(train,'sft_full.log')
        event('SFT_10K_COMPLETE',summary=json.loads((out/'summary.json').read_text()))
    except Exception as exc:
        event('BLOCKED',reason=str(exc))
        raise


if __name__=='__main__':main()
