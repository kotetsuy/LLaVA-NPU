"""YOLO11m NPU pre/post-processing (pure numpy + cv2, no torch/ultralytics).

This is the piece that replaces what Ultralytics did internally on the GPU
path: letterbox the input to 640, and decode ``output0`` [1,84,8400] back into
boxes *in the input frame's coordinate system*.

Kept deliberately dependency-light (numpy + cv2 only) so it imports cleanly in
the Ryzen AI venv (onnxruntime-vitisai) that the NPU sidecar runs under, which
does not have torch/ultralytics. Ported from ``~/yolotest/decode_detect.py``
with the letterbox *inverse* transform added — the yolotest script displayed
boxes in 640 space, which is only correct for near-square images.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


@dataclass(frozen=True)
class Letterbox:
    """Parameters of a letterbox needed to invert it.

    A source pixel maps to letterbox space as ``dst = src * scale + (left|top)``.
    So the inverse (letterbox coords -> source coords) is
    ``src = (dst - (left|top)) / scale``.
    """

    scale: float
    left: int
    top: int


def letterbox(frame_bgr: np.ndarray, size: int = 640) -> tuple[np.ndarray, Letterbox]:
    """Resize keeping aspect ratio and pad to ``size``x``size`` with 114 gray."""
    h, w = frame_bgr.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = cv2.resize(frame_bgr, (nw, nh))
    return canvas, Letterbox(scale=scale, left=left, top=top)


def preprocess(frame_bgr: np.ndarray, size: int = 640) -> tuple[np.ndarray, Letterbox]:
    """BGR frame -> (1,3,size,size) float32 RGB/255 tensor + letterbox params."""
    lb, params = letterbox(frame_bgr, size)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None])
    return tensor, params


def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy NMS. Returns indices to keep (into the passed arrays)."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[rest, 0])
        yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[rest, 1])
        xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[rest, 2])
        yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[rest, 3])
        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih
        area_i = (boxes_xyxy[i, 2] - boxes_xyxy[i, 0]) * (boxes_xyxy[i, 3] - boxes_xyxy[i, 1])
        area_r = (boxes_xyxy[rest, 2] - boxes_xyxy[rest, 0]) * (
            boxes_xyxy[rest, 3] - boxes_xyxy[rest, 1]
        )
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou < iou_thresh]
    return keep


def decode_detections(
    output0: np.ndarray,
    lb: Letterbox,
    frame_w: int,
    frame_h: int,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
) -> list[dict]:
    """Decode YOLO11 ``output0`` [1,84,8400] into boxes in *frame* coords.

    Returns a list of ``{"label","conf","cls","x1","y1","x2","y2"}`` dicts,
    x/y in the input frame's pixel coordinate system (e.g. 1280x720). NMS is
    class-aware (matches Ultralytics' default ``agnostic_nms=False``): boxes of
    different classes never suppress each other.
    """
    preds = np.asarray(output0)
    if preds.ndim == 3:
        preds = preds[0]  # (84, 8400)
    preds = preds.T  # (8400, 84)

    class_scores = preds[:, 4:]
    conf = class_scores.max(axis=1)
    cls = class_scores.argmax(axis=1)

    mask = conf > conf_thresh
    if not mask.any():
        return []

    boxes = preds[mask, :4]  # cx, cy, w, h  (letterbox 640 space)
    conf = conf[mask]
    cls = cls[mask]

    # cxcywh -> xyxy in letterbox space, then invert the letterbox to frame space.
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - bw / 2 - lb.left) / lb.scale
    y1 = (cy - bh / 2 - lb.top) / lb.scale
    x2 = (cx + bw / 2 - lb.left) / lb.scale
    y2 = (cy + bh / 2 - lb.top) / lb.scale
    xyxy = np.stack([x1, y1, x2, y2], axis=1)
    np.clip(xyxy[:, 0::2], 0, frame_w, out=xyxy[:, 0::2])
    np.clip(xyxy[:, 1::2], 0, frame_h, out=xyxy[:, 1::2])

    dets: list[dict] = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        for k in _nms(xyxy[idx], conf[idx], iou_thresh):
            j = idx[k]
            label = COCO_CLASSES[int(c)] if int(c) < len(COCO_CLASSES) else str(int(c))
            dets.append(
                {
                    "label": label,
                    "conf": round(float(conf[j]), 3),
                    "cls": int(c),
                    "x1": round(float(xyxy[j, 0]), 1),
                    "y1": round(float(xyxy[j, 1]), 1),
                    "x2": round(float(xyxy[j, 2]), 1),
                    "y2": round(float(xyxy[j, 3]), 1),
                }
            )
    return dets
