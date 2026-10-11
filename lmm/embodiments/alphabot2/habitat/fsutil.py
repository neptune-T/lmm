"""Writes that survive transient ENOMEM from the bosfs FUSE daemon (cgroup page cache near its limit:
writes through FUSE can fail with errno 12 instead of waiting for reclaim; observed killing workers)."""
import errno, os, sys, time

LOG_DIR = os.environ.get("ALPHABOT2_LOG_DIR", "/mnt/cfs/h0ddz1/personal/liyanjun/yanjun/lmm_runtime/logs")   # NFS, not bosfs


def _retry(fn, tries=30):
    for i in range(tries):
        try:
            return fn()
        except OSError as e:
            if e.errno != errno.ENOMEM or i == tries - 1:
                raise
            time.sleep(min(10.0, 0.5 * (i + 1)))


def append_line(path, line):
    """Open-append-close one line (visible immediately; retried on ENOMEM)."""
    def w():
        with open(path, "a") as f:
            f.write(line if line.endswith("\n") else line + "\n")
    _retry(w)


class RetryingStream:
    def __init__(self, s):
        self._s = s

    def write(self, x):
        return _retry(lambda: self._s.write(x))

    def flush(self):
        return _retry(self._s.flush)

    def __getattr__(self, k):
        return getattr(self._s, k)


def protect_stdio():
    sys.stdout = RetryingStream(sys.stdout)
    sys.stderr = RetryingStream(sys.stderr)
