"""Serve the public AlphaBot pi0.5 checkpoint over a local socket (length-prefixed pickle).

Request:  {"cmd": "infer", "images": {camera_key: HxWx3 uint8}, "state": 16 floats
           (checkpoint convention), "task": str, "seed": int}
Response: {"actions": (50, 16) float32, "latency_s": float}
Runs in the lerobot venv; the Isaac process is the client. Local use only (127.0.0.1).
"""
import os, pickle, socket, struct, sys, time, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pi05_runtime import Pi05Runtime  # noqa: E402

PORT = int(os.environ.get("PI05_PORT", "5577"))


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


def main():
    rt = Pi05Runtime()
    dummy = {k: np.zeros((240, 320, 3), np.uint8) for k in ("camera_front", "camera_left", "camera_right", "camera_top")}
    for _ in range(2):
        rt.infer(dummy, np.zeros(16), "warmup")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PORT))
    srv.listen(1)
    print(f"READY port={PORT}", flush=True)
    while True:
        conn, _ = srv.accept()
        with conn:
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    break
                try:
                    if msg.get("cmd") == "infer":
                        a, dt = rt.infer(msg["images"], msg["state"], msg["task"], int(msg.get("seed", 0)))
                        send_msg(conn, {"actions": a.astype(np.float32), "latency_s": dt})
                    elif msg.get("cmd") == "ping":
                        send_msg(conn, {"ok": True, "time": time.time()})
                    else:
                        send_msg(conn, {"error": f"unknown cmd {msg.get('cmd')!r}"})
                except Exception as e:  # report, keep serving
                    send_msg(conn, {"error": repr(e), "tb": traceback.format_exc()[-1500:]})


if __name__ == "__main__":
    main()
