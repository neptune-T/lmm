"""Read-only cgroup/GPU/heartbeat telemetry for this recovery run."""
import json
from pathlib import Path
import subprocess
import sys
import time
import shutil
import os

run=Path(sys.argv[1])
while True:
    row={'time':time.strftime('%FT%T%z')}
    row['storage']={}
    for name, path in [('root','/'),('run',str(run)),('shm','/dev/shm'),
                       ('hf_stage',str(run/'backups/hf_stage'))]:
        if not Path(path).exists(): continue
        usage=shutil.disk_usage(path); stat=os.statvfs(path)
        row['storage'][name]={'free_bytes':usage.free,'total_bytes':usage.total,'free_inodes':stat.f_favail}
    row['root_disk_warning']=row['storage']['root']['free_bytes'] < (8<<30)
    for key,filename in [('memory_current','memory.current'),('memory_peak','memory.peak'),('memory_limit','memory.max')]:
        p=Path('/sys/fs/cgroup')/filename
        if p.exists():
            v=p.read_text().strip()
            row[key]=int(v) if v.isdigit() else v
    p=Path('/sys/fs/cgroup/memory.events')
    if p.exists():row['memory_events']=p.read_text().strip()
    try:
        row['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    except Exception as exc:row['gpu_query_error']=type(exc).__name__
    p=run/'sft/v2B/heartbeat.json'
    if p.exists():
        try:row['training']=json.loads(p.read_text())
        except json.JSONDecodeError:pass
    print(json.dumps(row),flush=True)
    time.sleep(30)
