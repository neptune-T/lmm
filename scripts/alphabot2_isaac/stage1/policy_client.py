"""Client for scripts/alphabot2_pi05/policy_server.py (numpy-version-neutral codec)."""
import pickle, socket, struct, time

import numpy as np


def _enc(o):
    if isinstance(o, np.ndarray):
        return {"__nd__": True, "dtype": o.dtype.str, "shape": o.shape, "data": o.tobytes()}
    if isinstance(o, dict):
        return {k: _enc(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_enc(v) for v in o)
    if isinstance(o, np.generic):
        return o.item()
    return o


def _dec(o):
    if isinstance(o, dict):
        if o.get("__nd__"):
            return np.frombuffer(o["data"], dtype=np.dtype(o["dtype"])).reshape(o["shape"]).copy()
        return {k: _dec(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_dec(v) for v in o)
    return o


class PolicyClient:
    def __init__(self, port: int, host: str = "127.0.0.1", timeout: float = 900.0):
        for i in range(120):                      # the server may still be loading
            try:
                self.sock = socket.create_connection((host, port), timeout=timeout)
                break
            except OSError:
                time.sleep(5)
        else:
            raise RuntimeError(f"policy server on port {port} unreachable")
        self.info = self.rpc({"cmd": "ping"})

    def rpc(self, obj):
        b = pickle.dumps(_enc(obj), protocol=4)
        self.sock.sendall(struct.pack("!Q", len(b)) + b)
        n = struct.unpack("!Q", self.sock.recv(8, socket.MSG_WAITALL))[0]
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(min(1 << 20, n - len(buf)))
            if not chunk:
                raise RuntimeError("policy server closed the connection")
            buf.extend(chunk)
        out = _dec(pickle.loads(bytes(buf)))
        if "error" in out:
            raise RuntimeError(out["error"] + "\n" + out.get("tb", ""))
        return out

    def infer(self, images: dict, state16, task: str, seed: int):
        t = time.perf_counter()
        r = self.rpc({"cmd": "infer", "images": images, "state": np.asarray(state16, np.float32), "task": task,
                      "seed": int(seed)})
        r["roundtrip_s"] = time.perf_counter() - t
        return r
