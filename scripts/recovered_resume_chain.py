"""Resume a verified 1k reconstruction checkpoint after a container replacement.

All storage changes are recorded separately from the frozen training recipe.
Run in server tmux; no collection, RL, or extension of the 10k budget.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from backup_storage import checkpoint_environment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--session', default='lmm-recover')
    a = ap.parse_args()
    run = Path(a.run).resolve()
    code = Path(__file__).resolve().parent
    lock = open(run/'resume_chain.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def event(name, **kw):
        row = dict(time=time.strftime('%FT%T%z'), event=name, **kw)
        with open(run/'events.jsonl','a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    def receipt(path):
        d = json.loads(path.read_text())
        assert d['files'] > 0 and d['verified'] == d['files'] and not d['mismatch'], path
        return d

    def window(name, argv, env):
        names = subprocess.check_output(['tmux','list-windows','-t',a.session,'-F','#{window_name}'],text=True).splitlines()
        if name in names: raise RuntimeError(f'Window already exists: {name}; inspect before adopting')
        # Explicit env arguments are interpreted INSIDE the new tmux window.
        command = shlex.join(['env', *[f'{k}={v}' for k,v in env.items()], *argv])
        command += ' >> '+shlex.quote(str(run/'logs'/f'{name}.log'))+' 2>&1'
        subprocess.run(['tmux','new-window','-t',a.session,'-n',name,command],check=True)

    try:
        event('WAIT_RESTORED_CHECKPOINT_AND_HF')
        deadline=time.monotonic()+6*3600
        while not (run/'checks/checkpoint_copy_verified.json').exists():
            lines=(run/'restore_events.jsonl').read_text().splitlines()
            if lines and json.loads(lines[-1])['event']=='RESTORE_BLOCKED':
                raise RuntimeError('Restore blocked; inspect restore_events.jsonl')
            if time.monotonic()>deadline: raise RuntimeError('Restore wait timeout')
            time.sleep(15)
        inherited=run/'checks/inherited'
        for rel,key in [('data/recovered_integrity.json','pass'),('shard_alignment.json','passed'),('model_probe.json','passed')]:
            assert json.loads((inherited/rel).read_text())[key], rel
        receipt(inherited/'data_hf_verified.json')
        assert json.loads((run/'checks/data_copy_verified.json').read_text())['verified']
        ck=run/'sft/v2B/ckpt_step0001000'
        assert (ck/'COMPLETE').is_file()
        assert json.loads((run/'checks/checkpoint_copy_verified.json').read_text())['verified']
        assert (run/'checks/storage_tmux_test_pass.json').exists()
        assert json.loads((run/'checks/resume_tests_pass.json').read_text())['passed']
        env=checkpoint_environment(run)
        # No secret values in window arguments; credentials remain in external files.
        runtime_env={'CUDA_VISIBLE_DEVICES':'0,1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'2',
                     'TORCHDYNAMO_DISABLE':'1','NCCL_IB_DISABLE':'1','TOKENIZERS_PARALLELISM':'false',
                     'PI05_SPM':str(run/'assets/paligemma_tokenizer.model'),
                     'HF_HOME':str(run/'cache/hf'),'HF_XET_CACHE':str(run/'cache/xet'),
                     'TORCH_HOME':str(run/'cache/torch'),'XDG_CACHE_HOME':str(run/'cache/xdg'),
                     'WANDB_CACHE_DIR':str(run/'cache/wandb'),'PYTHONDONTWRITEBYTECODE':'1',
                     'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
        env.update(runtime_env)
        # Exercise the actual periodic worker, not only a direct preflight invocation.
        window('checkpoint-backup',[sys.executable,'-u',str(code/'recovered_checkpoint_backup.py'),
                                   '--run',str(run),'--watch'], {'PYTHONDONTWRITEBYTECODE':'1'})
        hf=run/'checks/checkpoint_backups/ckpt_step0001000_hf.json'
        while not hf.exists():
            panes=subprocess.check_output(['tmux','list-panes','-t',a.session+':checkpoint-backup','-F','#{pane_dead}'],text=True)
            if '1' in panes.split(): raise RuntimeError('Checkpoint uploader exited before HF verification')
            if time.monotonic()>deadline: raise RuntimeError('HF backup wait timeout')
            time.sleep(20)
        event('CHECKPOINT_1000_HF_VERIFIED',revision=receipt(hf)['revision'])
        assert shutil.disk_usage('/').free > 12*(1<<30),'root disk headroom'
        assert shutil.disk_usage(run).free > 600*(1<<30),'CFS headroom for remaining checkpoints and snapshots'
        for gpu in [0,1]:
            pids=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            assert not pids, f'GPU{gpu} occupied: {pids}'
        mapping={'out':{'from':'/share/personal/liyanjun/yanjun/zts/lmm/rebuild_20261027/sft/v2B','to':str(run/'sft/v2B')},
                 'init_checkpoint':{'from':'/share/personal/liyanjun/yanjun/zts/lmm/rebuild_20261027/assets/ckpt','to':str(run/'assets/ckpt')}}
        (run/'storage_map.json').write_text(json.dumps(mapping,indent=2))
        provenance={'owner':'zhangtianshan','parent_run':'rebuild_20261027','checkpoint_step':1000,
                    'storage_relocation':mapping,'training_config_unchanged':True,
                    'discarded_incomplete_step':2000,'cause':'HF tmux worker staged on root; ENOSPC, then validation failure/NCCL timeout'}
        (run/'checks/execution_provenance.json').write_text(json.dumps(provenance,indent=2))
        event('STRICT_RESUME_1000_TO_10000_START',world=2,global_batch=16)
        train=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
               str(code/'alphabot2_pi05/sft_ovmm.py'),'--config',str(run/'sft_v2B_ddp.yaml'),
               '--storage-map',str(run/'storage_map.json'),'--validate-resume']
        with open(run/'logs/sft_resumed.log','a') as log:
            result=subprocess.run(train,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'SFT exited {result.returncode}; inspect sft_resumed.log')
        assert (run/'sft/v2B/DONE').exists()
        event('SFT_10K_COMPLETE')
    except Exception as exc:
        event('BLOCKED',reason=str(exc))
        raise


if __name__=='__main__': main()
