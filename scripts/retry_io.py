"""Bounded retries for a restart-safe CPU command, never for simulation workers."""
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def transient_io_only(output):
    # A quality assertion or corrupt archive must never be hidden by a later atexit I/O error.
    if any(s in output for s in ('AssertionError', 'ValueError', 'BadZipFile', 'EOFError', 'CRC-32')):
        return False
    errors = re.findall(r'^([\w.]*Error): (.+)$', output, flags=re.MULTILINE)
    return bool(errors) and all(name == 'OSError' and re.match(r'\[Errno (5|70|116)\]', message)
                                for name, message in errors)


def run(command, attempts=12, sleep=time.sleep):
    # Logs remain on node-local /tmp even while the shared filesystem is inaccessible.
    logs = Path(tempfile.mkdtemp(prefix='lmm-cpu-io-'))
    print(f'CPU_IO_ATTEMPT_LOGS {logs}', flush=True)
    for attempt in range(1, attempts + 1):
        path = logs / f'attempt_{attempt:02d}.log'
        with path.open('w') as stream:
            rc = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
        output = path.read_text(errors='replace')
        print(output, end='', flush=True)
        if rc == 0:
            return 0
        if not transient_io_only(output) or attempt == attempts:
            print(f'CPU_IO_STOP attempt={attempt} rc={rc} evidence={path}', flush=True)
            return rc
        delay = min(300, 30 * 2 ** (attempt - 1))
        print(f'CPU_IO_RETRY attempt={attempt} wait_seconds={delay} evidence={path}', flush=True)
        sleep(delay)
    raise AssertionError('unreachable')
