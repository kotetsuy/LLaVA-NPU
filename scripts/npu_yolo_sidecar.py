#!/usr/bin/env python3
"""NPU YOLO11m sidecar — runs object detection on the AMD XDNA2 NPU.

This process is the NPU counterpart of ``src/inference/yolo_worker.py`` (which
runs Ultralytics on the GPU). It must run under the **Ryzen AI venv**, which is
the only environment that has ``onnxruntime-vitisai`` + XRT:

    source ~/ryzenai/ryzenai_venv/setup_ryzenai_env.sh
    PYTHONPATH=~/LLaVA-NPU python ~/LLaVA-NPU/scripts/npu_yolo_sidecar.py \
        --model models/yolo11m_a16w8.onnx --port 8082

Why a separate process (not a thread in ``serve``): the NPU stack needs
onnxruntime-vitisai and XRT's ``LD_LIBRARY_PATH``, which conflict with the
torch-ROCm / ultralytics venv the FastAPI server runs in. So we mirror the VLM
architecture: a standalone worker that ``serve`` talks to over loopback HTTP.

Flow: attach the capture SHM -> read the latest 1280x720 BGR frame -> letterbox
to 640 -> VitisAI EP inference on the NPU -> decode + class-aware NMS -> invert
the letterbox so boxes are back in 1280x720 -> publish as JSON on ``/latest``.
The bbox JSON schema matches ``YoloRunner._publish`` exactly, so the browser
overlay is byte-for-byte compatible with the GPU path.

Verify NPU offload while it's running (separate terminal):

    /opt/xilinx/xrt/bin/xrt-smi examine -d 0000:c6:00.1 -r all
    # HW Context = Active, Columns[0-7], Submissions increasing => on the NPU
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime as ort

from src.capture.shm_writer import FrameSHM
from src.npu_yolo.postprocess import decode_detections, preprocess

log = logging.getLogger("npu-yolo")


class _State:
    """Thread-safe holder for the latest bbox payload + readiness."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self.ready = threading.Event()

    def set(self, payload: dict) -> None:
        with self._lock:
            self._latest = payload

    def get(self) -> dict | None:
        with self._lock:
            return self._latest


def _build_session(model_path: str, use_cpu: bool) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = ["CPUExecutionProvider"] if use_cpu else ["VitisAIExecutionProvider"]
    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers, provider_options=[{}])
    log.info("onnxruntime session ready: providers=%s", sess.get_providers())
    return sess


def _attach_shm(shm_name: str, stop: threading.Event) -> FrameSHM | None:
    for _ in range(120):  # up to ~60s
        if stop.is_set():
            return None
        try:
            return FrameSHM.attach(shm_name)
        except (FileNotFoundError, RuntimeError):
            time.sleep(0.5)
    return None


def _inference_loop(
    sess: ort.InferenceSession,
    shm: FrameSHM,
    state: _State,
    stop: threading.Event,
    conf: float,
    iou: float,
    imgsz: int,
) -> None:
    input_name = sess.get_inputs()[0].name

    # Warm up so the first (slow) VitisAI subgraph compile happens before we
    # start publishing, and so /latest is populated promptly.
    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    t0 = time.time()
    sess.run(None, {input_name: dummy})
    log.info("warmup inference done in %.2fs", time.time() - t0)
    state.ready.set()

    last_seq = -1
    n, t_win = 0, time.time()
    while not stop.is_set():
        got = shm.read()
        if got is None:
            time.sleep(0.005)
            continue
        frame, meta = got
        if meta.seq == last_seq:
            time.sleep(0.001)
            continue
        last_seq = meta.seq

        if not meta.connected:
            # Capture is in SEARCHING (synthetic black); clear the overlay.
            state.set(_payload(meta, []))
            continue

        try:
            tensor, lb = preprocess(frame, imgsz)
            out = sess.run(None, {input_name: tensor})[0]
            boxes = decode_detections(out, lb, meta.frame_w, meta.frame_h, conf, iou)
        except Exception as e:  # noqa: BLE001
            log.warning("inference failed on seq=%d: %s", meta.seq, e)
            continue
        state.set(_payload(meta, boxes))

        n += 1
        if time.time() - t_win >= 5.0:
            log.info("throughput: %.1f inf/s (%d boxes last frame)", n / (time.time() - t_win), len(boxes))
            n, t_win = 0, time.time()

    log.info("inference loop exiting")


def _payload(meta, boxes: list[dict]) -> dict:
    return {
        "frame_seq": meta.seq,
        "ts_ns": meta.timestamp_ns,
        "frame_w": meta.frame_w,
        "frame_h": meta.frame_h,
        "connected": bool(meta.connected),
        "boxes": boxes,
    }


def _make_handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # silence per-request stderr spam
            return

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200 if state.ready.is_set() else 503, {"ready": state.ready.is_set()})
                return
            if self.path == "/latest":
                latest = state.get()
                if latest is None:
                    self._send(503, {"error": "no frame yet"})
                else:
                    self._send(200, latest)
                return
            self._send(404, {"error": "not found"})

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="path to yolo11m_a16w8.onnx")
    ap.add_argument("--shm", default="webcam_latest", help="capture SHM segment name")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--cpu", action="store_true", help="use CPU EP instead of NPU (debug)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    log.info("loading model %s (cpu=%s)", args.model, args.cpu)
    sess = _build_session(args.model, args.cpu)

    log.info("attaching SHM %r ...", args.shm)
    shm = _attach_shm(args.shm, stop)
    if shm is None:
        log.error("SHM %r never appeared (is capture-run up?); exiting", args.shm)
        return 1
    log.info("SHM attached: %dx%d %s", shm.frame_w, shm.frame_h, shm.pixel_format)

    state = _State()
    worker = threading.Thread(
        target=_inference_loop,
        args=(sess, shm, state, stop, args.conf, args.iou, args.imgsz),
        name="npu-yolo-infer",
        daemon=True,
    )
    worker.start()

    httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(state))
    httpd.timeout = 0.5
    log.info("serving bbox on http://%s:%d/latest", args.host, args.port)
    try:
        while not stop.is_set():
            httpd.handle_request()
    finally:
        log.info("shutting down")
        stop.set()
        worker.join(timeout=3.0)
        httpd.server_close()
        shm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
