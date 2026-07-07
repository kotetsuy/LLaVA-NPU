"""Background YOLO inference for the WebRTC server.

Two backends, selected by ``config.yaml`` ``yolo.backend``:

* ``gpu`` (default) — runs Ultralytics on the GPU *in this process*, in a
  daemon thread that reads the latest frame from SHM and runs
  ``YoloWorker.predict``. torch/ROCm inference releases the GIL during the C++
  kernel, so asyncio's event loop in the same process is not blocked.
* ``npu`` — inference happens in a **separate process** (``npu_yolo_sidecar``)
  running under the Ryzen AI venv on the XDNA2 NPU, because onnxruntime-vitisai
  + XRT can't share this venv. Here the thread just polls the sidecar's
  loopback HTTP ``/latest`` and republishes it, mirroring how ``VlmRunner``
  talks to llama-server.

Either way the public surface is identical: ``start`` / ``stop`` / ``ready`` /
``get_latest()`` returning the same bbox dict, so ``app.py`` is backend-agnostic.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_SIDECAR_URL = "http://127.0.0.1:8082"


class YoloRunner:
    def __init__(self, shm_name: str, yolo_cfg: dict[str, Any]) -> None:
        self._shm_name = shm_name
        self._yolo_cfg = yolo_cfg
        self._backend = str(yolo_cfg.get("backend", "gpu")).lower()
        self._latest: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="yolo-runner", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=join_timeout)
        if self._thread.is_alive():
            log.warning("yolo-runner thread did not exit within %.1fs", join_timeout)

    def get_latest(self) -> dict[str, Any] | None:
        with self._lock:
            return self._latest

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def _run(self) -> None:
        if self._backend == "npu":
            self._run_npu()
        else:
            self._run_gpu()

    def _run_npu(self) -> None:
        """Poll the NPU sidecar's loopback HTTP and republish its bbox JSON.

        The sidecar (``scripts/npu_yolo_sidecar.py``, Ryzen AI venv) already
        emits the exact ``_publish`` schema, so we forward it verbatim. It may
        not be up yet when ``serve`` starts — we retry until it answers.
        """
        import requests  # noqa: PLC0415

        npu_cfg = self._yolo_cfg.get("npu", {})
        base = str(npu_cfg.get("sidecar_url", _DEFAULT_SIDECAR_URL)).rstrip("/")
        url = f"{base}/latest"
        poll_period = 1.0 / float(npu_cfg.get("poll_hz", 60))
        log.info("yolo-runner[npu]: polling sidecar %s at %.0f Hz", url, 1.0 / poll_period)

        session = requests.Session()
        warned = False
        while not self._stop.is_set():
            try:
                resp = session.get(url, timeout=1.0)
                if resp.status_code == 200:
                    with self._lock:
                        self._latest = resp.json()
                    if not self._ready.is_set():
                        log.info("yolo-runner[npu]: sidecar ready")
                        self._ready.set()
                    warned = False
                elif resp.status_code == 503:
                    # sidecar up but no frame yet — keep waiting quietly
                    pass
            except requests.RequestException as e:
                if not warned:
                    log.info("yolo-runner[npu]: sidecar not reachable yet (%s); retrying", e)
                    warned = True
                self._stop.wait(0.5)
                continue
            self._stop.wait(poll_period)
        log.info("yolo-runner[npu]: thread exited")

    def _run_gpu(self) -> None:
        # Defer heavy imports to the thread so server startup stays snappy.
        from src.capture.shm_writer import FrameSHM  # noqa: PLC0415
        from src.inference.yolo_worker import YoloWorker  # noqa: PLC0415

        log.info("yolo-runner: loading model %s (device=%s)", self._yolo_cfg["model"], self._yolo_cfg["device"])
        try:
            worker = YoloWorker(
                model_path=self._yolo_cfg["model"],
                device=self._yolo_cfg["device"],
                imgsz=self._yolo_cfg["imgsz"],
                conf=self._yolo_cfg["conf"],
                iou=self._yolo_cfg["iou"],
                half=self._yolo_cfg.get("half", False),
            )
        except Exception:
            log.exception("yolo-runner: model load failed; thread exiting")
            return

        # Wait up to 30 s for capture-run to create the SHM segment.
        shm: FrameSHM | None = None
        for _ in range(60):
            if self._stop.is_set():
                return
            try:
                shm = FrameSHM.attach(self._shm_name)
                break
            except (FileNotFoundError, RuntimeError):
                time.sleep(0.5)
        if shm is None:
            log.error("yolo-runner: SHM %r never appeared; thread exiting", self._shm_name)
            return

        log.info("yolo-runner: ready")
        self._ready.set()
        last_seq = -1
        try:
            while not self._stop.is_set():
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
                    # capture is in SEARCHING; emit empty boxes so client clears overlay
                    self._publish(meta, [])
                    continue
                try:
                    dets = worker.predict(frame)
                except Exception as e:  # noqa: BLE001
                    log.warning("yolo-runner: predict raised %s", e)
                    continue
                self._publish(meta, dets)
        finally:
            shm.close()
            log.info("yolo-runner: thread exited")

    def _publish(self, meta, dets) -> None:
        boxes = [
            {
                "label": d.label,
                "conf": round(d.conf, 3),
                "x1": round(d.xyxy[0], 1),
                "y1": round(d.xyxy[1], 1),
                "x2": round(d.xyxy[2], 1),
                "y2": round(d.xyxy[3], 1),
            }
            for d in dets
        ]
        with self._lock:
            self._latest = {
                "frame_seq": meta.seq,
                "ts_ns": meta.timestamp_ns,
                "frame_w": meta.frame_w,
                "frame_h": meta.frame_h,
                "connected": meta.connected,
                "boxes": boxes,
            }
