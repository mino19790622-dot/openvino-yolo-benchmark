"""Unit tests for the torch-free pre/post-processing maths in ``yolo_utils.py``.

These cover the part of the pipeline that decides *where* boxes land and *which*
boxes survive, so they need neither OpenVINO nor a model file. The numbers are
hand-computed from the geometry rather than copied from a previous run, so a
regression in the maths fails here instead of silently shifting every box.
"""

import numpy as np
import pytest

from yolo_utils import (
    _iou_matrix,
    _needs_sigmoid,
    compute_coco_map,
    decode,
    letterbox,
    multiclass_nms,
    nms,
    preprocess,
)

NUM_BOXES = 100  # must exceed NUM_CLASSES so decode's layout heuristic is exercised


# ------------------------------------------------------------------ letterbox
def test_letterbox_keeps_aspect_ratio_and_centers():
    im = np.full((100, 200, 3), 7, dtype=np.uint8)
    canvas, ratio, pad = letterbox(im, 640)
    assert canvas.shape == (640, 640, 3)
    assert canvas.dtype == np.uint8
    # min(640/100, 640/200) = 3.2 -> a 320x640 strip, centred vertically
    assert ratio == pytest.approx(3.2)
    assert pad == (0, 160)


def test_letterbox_pads_with_grey_114():
    im = np.full((100, 200, 3), 7, dtype=np.uint8)
    canvas, _, (left, top) = letterbox(im, 640)
    assert canvas[0, 0].tolist() == [114, 114, 114]
    assert canvas[639, 639].tolist() == [114, 114, 114]
    assert canvas[top, left].tolist() == [7, 7, 7]


# ----------------------------------------------------------------- preprocess
def test_preprocess_shape_dtype_and_scale():
    im = np.full((100, 200, 3), 255, dtype=np.uint8)
    x, _, _ = preprocess(im, 640)
    assert x.shape == (1, 3, 640, 640)
    assert x.dtype == np.float32
    assert float(x.max()) == pytest.approx(1.0)
    assert float(x.min()) == pytest.approx(114 / 255)


def test_preprocess_converts_bgr_to_rgb():
    im = np.zeros((100, 200, 3), dtype=np.uint8)
    im[:, :] = (10, 20, 30)  # BGR
    x, _, (left, top) = preprocess(im, 640)
    assert float(x[0, 0, top, left]) == pytest.approx(30 / 255)  # R
    assert float(x[0, 1, top, left]) == pytest.approx(20 / 255)  # G
    assert float(x[0, 2, top, left]) == pytest.approx(10 / 255)  # B


# ------------------------------------------------------------------------ NMS
def test_nms_suppresses_the_lower_scoring_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    # IoU(box0, box1) = 81/119 = 0.68 > 0.5 -> box1 is suppressed
    assert nms(boxes, scores, iou_thr=0.5).tolist() == [0, 2]


def test_nms_on_empty_input_returns_empty():
    keep = nms(np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32), 0.5)
    assert keep.size == 0


def test_multiclass_nms_does_not_suppress_across_classes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    labels = np.array([1, 2])
    out_b, _, out_l = multiclass_nms(boxes, scores, labels, iou_thr=0.5)
    assert len(out_b) == 2, "identical boxes of different classes must both survive"
    assert sorted(out_l.tolist()) == [1, 2]


def test_multiclass_nms_respects_max_det_and_keeps_top_scores():
    boxes = np.array([[i * 20, 0, i * 20 + 5, 5] for i in range(10)], dtype=np.float32)
    scores = np.linspace(0.1, 1.0, 10).astype(np.float32)
    labels = np.zeros(10, dtype=int)
    out_b, out_s, _ = multiclass_nms(boxes, scores, labels, iou_thr=0.5, max_det=3)
    assert len(out_b) == 3
    assert out_s[0] >= out_s[1] >= out_s[2]


# -------------------------------------------------------------------- IoU / σ
def test_iou_matrix_identity_and_disjoint():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    m = _iou_matrix(a, b)
    assert m.shape == (1, 2)
    assert float(m[0, 0]) == pytest.approx(1.0)
    assert float(m[0, 1]) == pytest.approx(0.0)


def test_needs_sigmoid_distinguishes_logits_from_probabilities():
    assert _needs_sigmoid(np.array([[0.1, 0.9]])) is False
    assert _needs_sigmoid(np.array([[0.1, 2.5]])) is True
    assert _needs_sigmoid(np.array([[-1.0, 0.5]])) is True


# ---------------------------------------------------------------------- decode
def _pred(cx, cy, w, h, cls, score, num_classes=80, num_boxes=NUM_BOXES, transposed=False):
    """One confident box among ``num_boxes`` zeros, in either ONNX layout."""
    if transposed:
        pred = np.zeros((1, num_boxes, 4 + num_classes), dtype=np.float32)
        pred[0, 0, :4] = (cx, cy, w, h)
        pred[0, 0, 4 + cls] = score
    else:
        pred = np.zeros((1, 4 + num_classes, num_boxes), dtype=np.float32)
        pred[0, :4, 0] = (cx, cy, w, h)
        pred[0, 4 + cls, 0] = score
    return pred


def test_decode_accepts_the_channels_first_layout():
    boxes, scores, labels = decode(
        _pred(320, 320, 100, 100, cls=5, score=0.9),
        ratio=1.0,
        pad=(0, 0),
        orig_wh=(640, 640),
    )
    assert len(boxes) == 1
    assert boxes[0].tolist() == pytest.approx([270, 270, 370, 370])
    assert labels.tolist() == [5]
    assert float(scores[0]) == pytest.approx(0.9)


def test_decode_accepts_the_transposed_layout():
    boxes, _, labels = decode(
        _pred(320, 320, 100, 100, cls=3, score=0.8, transposed=True),
        ratio=1.0,
        pad=(0, 0),
        orig_wh=(640, 640),
    )
    assert len(boxes) == 1
    assert labels.tolist() == [3]


def test_decode_undoes_letterbox_padding_and_scale():
    # content was scaled 2x and pasted at (100, 50) on the canvas
    boxes, _, _ = decode(
        _pred(320, 320, 64, 64, cls=1, score=0.9),
        ratio=2.0,
        pad=(100, 50),
        orig_wh=(1000, 1000),
    )
    assert boxes[0].tolist() == pytest.approx([94, 119, 126, 151])


def test_decode_applies_sigmoid_when_the_output_is_logits():
    _, scores, _ = decode(
        _pred(320, 320, 100, 100, cls=5, score=2.0),
        ratio=1.0,
        pad=(0, 0),
        orig_wh=(640, 640),
    )
    assert float(scores[0]) == pytest.approx(1 / (1 + np.exp(-2.0)))


def test_decode_drops_boxes_below_the_confidence_threshold():
    boxes, _, _ = decode(
        _pred(320, 320, 100, 100, cls=5, score=0.4),
        ratio=1.0,
        pad=(0, 0),
        orig_wh=(640, 640),
        conf_thr=0.5,
    )
    assert len(boxes) == 0


def test_decode_clips_to_the_original_image_bounds():
    # a box that runs off the right/bottom edge must come back clipped
    boxes, _, _ = decode(
        _pred(630, 630, 100, 100, cls=5, score=0.9),
        ratio=1.0,
        pad=(0, 0),
        orig_wh=(640, 640),
    )
    assert boxes[0].tolist() == pytest.approx([580, 580, 640, 640])


# ------------------------------------------------------------------------- mAP
def _one_image(gt_boxes, gt_classes, boxes, scores, labels):
    """One prediction record, shaped the way ``compute_coco_map`` expects it."""
    return {
        "gt_boxes": np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4),
        "gt_classes": np.asarray(gt_classes, dtype=int),
        "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        "scores": np.asarray(scores, dtype=np.float32),
        "labels": np.asarray(labels, dtype=int),
    }


def test_map_is_one_for_a_perfect_detection():
    img = _one_image([[0, 0, 10, 10]], [5], [[0, 0, 10, 10]], [0.9], [5])
    res = compute_coco_map([img])
    assert res["mAP50"] == pytest.approx(1.0)
    assert res["mAP50-95"] == pytest.approx(1.0)
    # only the class that actually carries ground truth gets scored
    assert res["classes_evaluated"] == 1


def test_map_is_zero_when_the_only_detection_misses_every_gt():
    img = _one_image([[0, 0, 10, 10]], [5], [[500, 500, 510, 510]], [0.9], [5])
    res = compute_coco_map([img])
    assert res["mAP50"] == pytest.approx(0.0)
    assert res["mAP50-95"] == pytest.approx(0.0)


def test_map_averages_over_the_ten_coco_iou_thresholds():
    # IoU(box, gt) = 80/100 = 0.8, so it counts at 0.50..0.80 but not 0.85..0.95:
    # 7 of the 10 thresholds -> mAP50 stays 1.0 while mAP50-95 drops to 7/10
    img = _one_image([[0, 0, 10, 10]], [5], [[0, 0, 8, 10]], [0.9], [5])
    res = compute_coco_map([img])
    assert res["mAP50"] == pytest.approx(1.0)
    assert res["mAP50-95"] == pytest.approx(0.7)


def test_map_uses_101_point_interpolation_for_partial_recall():
    # 2 GT boxes but 1 detection -> recall caps at 0.5, so only the 51 thresholds
    # t in {0.00, 0.01, ..., 0.50} retain precision 1.0 -> AP = 51/101
    img = _one_image(
        [[0, 0, 10, 10], [100, 100, 110, 110]], [5, 5], [[0, 0, 10, 10]], [0.9], [5]
    )
    res = compute_coco_map([img])
    assert res["mAP50"] == pytest.approx(51 / 101)


def test_map_scores_each_class_that_has_ground_truth():
    img = _one_image(
        [[0, 0, 10, 10], [100, 100, 110, 110]],
        [5, 7],
        [[0, 0, 10, 10], [100, 100, 110, 110]],
        [0.9, 0.8],
        [5, 7],
    )
    res = compute_coco_map([img])
    assert res["mAP50"] == pytest.approx(1.0)
    assert res["classes_evaluated"] == 2


def test_map_without_any_detection_reports_no_evaluated_class():
    img = _one_image([[0, 0, 10, 10]], [5], [], [], [])
    res = compute_coco_map([img])
    assert res["classes_evaluated"] == 0
    assert np.isnan(res["mAP50"])
    # the class is still recorded, but with no AP attached
    assert res["per_class"][5]["npos"] == 1
    assert res["per_class"][5]["map"] is None
