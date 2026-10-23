"""Restart-safe CPU raw audit with bounded shared-filesystem I/O recovery.

Only explicit transient OSError 5/70/116 is retried. Simulation is never rerun.
Quality assertions and corrupt data fail immediately with the original evidence.
"""
from pathlib import Path
import sys
from retry_io import run

if __name__ == '__main__':
    raise SystemExit(run([sys.executable, str(Path(__file__).with_name('raw_history_audit.py')), *sys.argv[1:]]))
