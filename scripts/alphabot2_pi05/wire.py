"""Local policy-socket wire format (length-prefixed pickle, numpy-version-neutral). numpy only, so the
Habitat (python 3.9) client and the lerobot server (python 3.12) can both import it."""
import pickle, socket, struct

import numpy as np


def _enc(o):
    """numpy-version-neutral encoding (numpy 2 pickles reference numpy._core)."""
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


def recv_msg(conn):
    head = conn.recv(8, socket.MSG_WAITALL)
    if len(head) < 8:
        return None
    n = struct.unpack("!Q", head)[0]
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            return None
        buf.extend(chunk)
    return _dec(pickle.loads(bytes(buf)))


def send_msg(conn, obj):
    b = pickle.dumps(_enc(obj), protocol=4)
    conn.sendall(struct.pack("!Q", len(b)) + b)
