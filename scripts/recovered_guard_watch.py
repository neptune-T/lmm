"""Observe/restart the existing owned GPU guard supervisor, never training.

Does not create CUDA contexts or terminate processes. The existing supervisor's
worker enforces GPU0 exclusivity and yields to training. Container loss cannot
be recovered by any process inside that same container.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time


def supervisor_alive(root):
    with open(root/'guard.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runtime',default='/root/lmm-runtime')
    ap.add_argument('--run',required=True)
    a=ap.parse_args(); root=Path(a.runtime); run=Path(a.run)
    lock=open(root/'guard_watch.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=None; last_start=0
    while not (root/'STOP').exists():
        alive=supervisor_alive(root)
        if not alive and time.monotonic()-last_start>10:
            with open(root/'gpu0_guard.log','a') as log:
                child=subprocess.Popen([sys.executable,'-u',str(root/'gpu0_guard.py')],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            last_start=time.monotonic()
        if child is not None:child.poll()
        try:
            pids=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],text=True,timeout=5).split()
            row={'time':time.strftime('%FT%T%z'),'supervisor_alive':alive,
                 'gpu0_pids':pids,'state':'WAIT_GPU_RELEASE' if pids else 'GPU_FREE_GUARD_QUEUED',
                 'training_heartbeat':json.loads((run/'sft/v2B/heartbeat.json').read_text())}
            (run/'checks/guard_watch_status.json').write_text(json.dumps(row)+'\n')
        except Exception as exc:
            print(time.strftime('%FT%T%z'),'OBSERVATION_ERROR',type(exc).__name__,flush=True)
        time.sleep(2)


if __name__=='__main__':main()
