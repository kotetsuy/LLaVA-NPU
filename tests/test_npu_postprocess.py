"""Pure-logic tests for the NPU YOLO pre/post-processing.

The riskiest bit of the NPU path is the *inverse* letterbox: boxes come out of
the model in 640x640 letterbox space and must land back in the input frame's
coordinate system (typically 1280x720). These tests pin that transform with a
hand-constructed ``output0`` so a regression can't silently shift every box.
"""

from __future__ import annotations

import numpy as np

from src.npu_yolo.postprocess import decode_detections, letterbox, preprocess


def test_letterbox_params_for_16x9():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas, lb = letterbox(frame, 640)
    assert canvas.shape == (640, 640, 3)
    # 1280 is the long side: scale = 640/1280 = 0.5, width fills, height padded.
    assert lb.scale == 0.5
    assert lb.left == 0
    assert lb.top == (640 - 360) // 2 == 140


def test_preprocess_tensor_shape_and_range():
    frame = np.full((720, 1280, 3), 255, dtype=np.uint8)
    tensor, _ = preprocess(frame, 640)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0


def _one_box_output(cx, cy, w, h, cls_id, score, n_anchors=8400, n_cls=80):
    """Build an [1,84,8400] tensor with a single above-threshold detection."""
    out = np.zeros((1, 4 + n_cls, n_anchors), dtype=np.float32)
    out[0, 0, 0] = cx
    out[0, 1, 0] = cy
    out[0, 2, 0] = w
    out[0, 3, 0] = h
    out[0, 4 + cls_id, 0] = score
    return out


def test_decode_inverts_letterbox_to_frame_coords():
    # A box centered in 640 space at (320, 320) with size 100x100.
    out = _one_box_output(cx=320, cy=320, w=100, h=100, cls_id=0, score=0.9)
    lb = preprocess(np.zeros((720, 1280, 3), dtype=np.uint8), 640)[1]  # scale .5, top 140

    dets = decode_detections(out, lb, frame_w=1280, frame_h=720, conf_thresh=0.25)
    assert len(dets) == 1
    d = dets[0]
    assert d["label"] == "person"
    assert d["conf"] == 0.9
    # scale 0.5 => the 100px box becomes 200px in frame space.
    # x: (320 -/+ 50 - 0)  / 0.5 = 540 / 740
    # y: (320 -/+ 50 - 140)/ 0.5 = 260 / 460
    assert d["x1"] == 540.0 and d["x2"] == 740.0
    assert d["y1"] == 260.0 and d["y2"] == 460.0


def test_decode_clips_to_frame_bounds():
    # Box pushed past the right/bottom edges must be clamped to the frame.
    out = _one_box_output(cx=630, cy=630, w=400, h=400, cls_id=2, score=0.8)
    lb = preprocess(np.zeros((720, 1280, 3), dtype=np.uint8), 640)[1]
    dets = decode_detections(out, lb, frame_w=1280, frame_h=720, conf_thresh=0.25)
    assert len(dets) == 1
    d = dets[0]
    assert 0.0 <= d["x1"] and d["x2"] <= 1280.0
    assert 0.0 <= d["y1"] and d["y2"] <= 720.0


def test_decode_below_threshold_returns_empty():
    out = _one_box_output(cx=320, cy=320, w=100, h=100, cls_id=0, score=0.1)
    lb = preprocess(np.zeros((720, 1280, 3), dtype=np.uint8), 640)[1]
    assert decode_detections(out, lb, 1280, 720, conf_thresh=0.25) == []


def test_class_aware_nms_keeps_overlapping_different_classes():
    # Two heavily-overlapping boxes of *different* classes must both survive.
    out = np.zeros((1, 84, 8400), dtype=np.float32)
    for anchor, cls_id in ((0, 0), (1, 2)):
        out[0, 0, anchor] = 320
        out[0, 1, anchor] = 320
        out[0, 2, anchor] = 100
        out[0, 3, anchor] = 100
        out[0, 4 + cls_id, anchor] = 0.9
    lb = preprocess(np.zeros((720, 1280, 3), dtype=np.uint8), 640)[1]
    dets = decode_detections(out, lb, 1280, 720, conf_thresh=0.25, iou_thresh=0.45)
    assert {d["cls"] for d in dets} == {0, 2}
