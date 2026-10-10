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
from wire import _dec, _enc, recv_msg, send_msg  # noqa: E402,F401

PORT = int(os.environ.get("PI05_PORT", "5577"))


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
