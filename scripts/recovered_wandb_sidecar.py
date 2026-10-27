"""Read-only W&B logger for recovered v2-B; runs outside the training environment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def flatten(obj,prefix=''):
    out={}
    for key,value in obj.items():
        name=prefix+key
        if isinstance(value,dict):out.update(flatten(value,name+'/'))
        elif isinstance(value,(int,float)):out[name]=value
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True)
    ap.add_argument('--key-file',default='/root/.config/lmm/wandb_api_key')
    ap.add_argument('--entity',default='2980976495-peking-university')
    ap.add_argument('--project',default='VLA-baseline')
    a=ap.parse_args(); root=Path(a.run); out=root/'sft/v2B'
    lock=open(root/'wandb_sidecar.lock','a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ['WANDB_API_KEY']=Path(a.key_file).read_text().strip()
    os.environ['WANDB_DIR']=str(root/'wandb')
    os.environ['WANDB_SILENT']='true'
    os.environ['WANDB_DISABLE_CODE']='true'
    Path(os.environ['WANDB_DIR']).mkdir(exist_ok=True)
    import wandb
    info=json.loads((out/'run_info_step000000.json').read_text())
    run=wandb.init(entity=a.entity,project=a.project,id='alphabot2-v2B-recovered-ddp2-20261027',
        name='AlphaBot2-OVMM-v2-B-Recovered-DDP2-20261027',resume='allow',job_type='sft',
        tags=['recovered','v2-B','PA-OG-evaluation','dual-A800'],
        config={'code_commit':'2fdb084','dataset_version':'v2-9a54b0ff11a4',
                'original_v2B_checkpoint_resumed':False,'original_manifest_file_found':False,
                'eval_protocol':'PA-OG + Base Lock; observable switches; FP32 max-batch=1',
                'training_run_info':info})
    run.define_metric('trainer/step')
    run.define_metric('*',step_metric='trainer/step')
    statefile=root/'checks/wandb_state.json'
    state=json.loads(statefile.read_text()) if statefile.exists() else {'lines':0,'saved':[],'backed':[],'last_step':0}
    status=root/'checks/wandb_status.json'
    while True:
        lines=(out/'train_log.jsonl').read_text().splitlines()
        for line in lines[state['lines']:]:
            try:record=json.loads(line)
            except json.JSONDecodeError:break
            step=record['step']; state['last_step']=step
            prefix='val/' if 'val_loss' in record else 'train/'
            metrics=flatten({k:v for k,v in record.items() if k!='step'},prefix)
            run.log({'trainer/step':step,**metrics})
            state['lines']+=1
        try:
            text=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used',
                '--format=csv,noheader,nounits'],text=True,timeout=10)
            metrics={}
            for line in text.splitlines():
                idx,util,mem=line.split(',')
                if int(idx) in (0,1):
                    metrics[f'sys/gpu{int(idx)}/util_pct']=float(util)
                    metrics[f'sys/gpu{int(idx)}/memory_mib']=float(mem)
            run.log({'trainer/step':state['last_step'],**metrics})
        except (subprocess.SubprocessError,ValueError):pass
        for path in sorted(out.glob('ckpt_step*/COMPLETE')):
            step=int(path.parent.name.removeprefix('ckpt_step'))
            if step not in state['saved']:
                run.log({'trainer/step':state['last_step'],'ckpt/saved_step':step});state['saved'].append(step)
        for path in sorted((root/'checks/checkpoint_backups').glob('ckpt_step*_hf.json')):
            receipt=json.loads(path.read_text())
            if receipt['mismatch'] or receipt['verified']!=receipt['files']:continue
            step=int(path.name.removeprefix('ckpt_step').removesuffix('_hf.json'))
            if step not in state['backed']:
                run.log({'trainer/step':state['last_step'],'ckpt/hf_verified_step':step});state['backed'].append(step)
        if (root/'events.jsonl').exists():
            event=json.loads((root/'events.jsonl').read_text().splitlines()[-1])
            run.summary['driver_event']=event['event']
        statefile.write_text(json.dumps(state)+'\n')
        status.write_text(json.dumps({'time':time.strftime('%FT%T%z'),'url':run.url,
                                     'last_step':state['last_step'],'logged_records':state['lines']})+'\n')
        if (out/'DONE').exists():
            run.summary.update(json.loads((out/'summary.json').read_text()))
            if (root/'checks/final_training_hf_verified.json').exists():
                run.summary['final_hf_verified']=True;run.finish();return
        time.sleep(30)


if __name__=='__main__':main()
