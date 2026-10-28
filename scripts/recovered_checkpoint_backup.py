"""Back up immutable COMPLETE checkpoints to verified HF.
Optional mirrors must be explicitly requested; no implicit BOS writes.
Never removes originals; best/last are represented by the saved checkpoint step.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from recover_v2_dataset import digest, write
from backup_storage import checkpoint_environment


def backup(run, ckpt, mirror_root=None):
    env = checkpoint_environment(run)
    assert (ckpt/'COMPLETE').is_file()
    receipts=run/'checks/checkpoint_backups'
    receipts.mkdir(parents=True,exist_ok=True)
    hf=receipts/(ckpt.name+'_hf.json')
    if hf.exists():
        rec=json.loads(hf.read_text())
        assert not rec['mismatch'] and rec['verified']==rec['files']
        return
    if mirror_root is not None:
        dst=Path(mirror_root)/ckpt.name
        manifest={}
        for src in sorted(ckpt.rglob('*')):
            if not src.is_file() or src.name=='COMPLETE':continue
            assert not src.is_symlink()
            rel=src.relative_to(ckpt)
            sha=digest(src)
            target=dst/rel
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists() or digest(target)!=sha:
                temp=target.with_suffix(target.suffix+'.part')
                shutil.copyfile(src,temp)
                assert digest(temp)==sha,target
                temp.replace(target)
            manifest[str(rel)]={'bytes':src.stat().st_size,'sha256':sha}
        write(receipts/(ckpt.name+'_mounted.json'),dict(verified=True,source=str(ckpt),destination=str(dst),files=manifest))
        (dst/'COMPLETE').write_text('Verified file-by-file; see mounted receipt.\n')
        print('MOUNTED_BACKUP_VERIFIED',ckpt.name,flush=True)
    cmd=[sys.executable,str(Path(__file__).with_name('hf_sync.py')),
         '--tag',run.name+'_'+ckpt.name,'--receipt',str(hf),'--cleanup-verified-stage',str(ckpt)]
    subprocess.run(cmd,check=True,env=env)
    print('HF_CHECKPOINT_VERIFIED',ckpt.name,flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True)
    ap.add_argument('--checkpoint')
    ap.add_argument('--watch',action='store_true')
    ap.add_argument('--after-training',action='store_true',help='Do not upload until the run has DONE')
    ap.add_argument('--mirror-root',help='Optional explicit persistent mirror; omitted means no extra filesystem copies')
    a=ap.parse_args()
    run=Path(a.run)
    if a.after_training:
        print('WAIT_TRAINING_DONE_FOR_HF',flush=True)
        while not (run/'sft/v2B/DONE').exists():time.sleep(30)
    if a.checkpoint:
        backup(run,Path(a.checkpoint),a.mirror_root)
        return
    while True:
        for ck in sorted((run/'sft/v2B').glob('ckpt_step*')):
            if (ck/'COMPLETE').exists():backup(run,ck,a.mirror_root)
        if (run/'sft/v2B/DONE').exists():
            out=run/'sft/v2B'
            summary=json.loads((out/'summary.json').read_text())
            index=run/'checks/checkpoint_index.json'
            write(index,dict(best_step=summary['best_step'],
                            best_checkpoint=f"ckpt_step{summary['best_step']:07d}",
                            final_checkpoint=f"ckpt_step{summary['steps']:07d}",
                            selection=summary['selection']))
            receipt=run/'checks/final_training_hf_verified.json'
            if not receipt.exists():
                subprocess.run([sys.executable,str(Path(__file__).with_name('hf_sync.py')),
                    '--tag',run.name+'_final_results','--receipt',str(receipt),
                    str(out/'summary.json'),str(out/'train_log.jsonl'),str(index),
                    str(run/'sft_v2B_ddp.yaml'),str(run/'checks/execution_provenance.json'),
                    *map(str,sorted(out.glob('run_info_step*.json')))],check=True,env=checkpoint_environment(run))
            print('HF_FINAL_TRAINING_RESULTS_VERIFIED',flush=True)
            return
        if not a.watch:return
        time.sleep(30)


if __name__=='__main__':main()
