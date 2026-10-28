"""Keep immutable HF snapshots off ephemeral root storage."""
import os
from pathlib import Path
import shutil


def checkpoint_environment(run):
    """Set inside the worker: tmux does not forward an arbitrary caller env."""
    run = Path(run).resolve()
    stage = run / 'backups/hf_stage'
    stage.mkdir(parents=True, exist_ok=True)
    if stage.stat().st_dev == Path('/').stat().st_dev:
        raise RuntimeError('Checkpoint HF staging must be on a separate persistent mount')
    env = dict(os.environ)
    env.update(HF_SYNC_STAGE=str(stage), HF_HOME=str(run / 'cache/hf'),
               HF_XET_CACHE=str(run / 'cache/xet'), HF_UPLOAD_WORKERS='2')
    return env


def check_snapshot_capacity(stage, total_bytes):
    stage = Path(stage).resolve()
    stage.mkdir(parents=True, exist_ok=True)
    if total_bytes >= (1 << 30) and stage.stat().st_dev == Path('/').stat().st_dev:
        raise RuntimeError('Refusing >=1GiB HF snapshot on root filesystem; set HF_SYNC_STAGE to persistent storage')
    # Include a second copy if hardlinks are unavailable, plus operating headroom.
    required = 2 * total_bytes + (5 << 30)
    if shutil.disk_usage(stage).free < required:
        raise RuntimeError(f'Insufficient HF staging space: require {required} bytes')
    return required
