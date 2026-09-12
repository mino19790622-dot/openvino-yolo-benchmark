"""YOLOv8 前后处理与 COCO mAP 计算 —— 纯 numpy 实现，不依赖 PyTorch。

这台机器是 macOS x86_64，PyTorch 没有对应 wheel（官方只提供 arm64），
所以整条链路用 OpenVINO + numpy 完成，反而让项目本身更轻、更好复现。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = 640
NUM_CLASSES = 80


# --------------------------------------------------------------------------
# 前处理：letterbox + HWC(BGR) -> NCHW(RGB) + /255
# --------------------------------------------------------------------------
def letterbox(im: np.ndarray, size: int = IMG_SIZE):
    """保持长宽比缩放并居中填充，返回画布、缩放比、填充量。"""
    h, w = im.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = im
    return canvas, r, (left, top)


def preprocess(im: np.ndarray, size: int = IMG_SIZE):
    canvas, r, pad = letterbox(im, size)
    x = canvas[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return x[None, ...], r, pad


# --------------------------------------------------------------------------
# NMS
# --------------------------------------------------------------------------
def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """单类 NMS，返回保留下来的下标。"""
    if boxes.size == 0:
        return np.empty(0, dtype=int)
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=int)


def multiclass_nms(boxes, scores, labels, conf_thr=0.001, iou_thr=0.7, max_det=300):
    """按类别分别做 NMS（与 ultralytics 默认行为一致）。"""
    out_b, out_s, out_l = [], [], []
    for c in np.unique(labels):
        m = labels == c
        b, s = boxes[m], scores[m]
        k = nms(b, s, iou_thr)
        out_b.append(b[k])
        out_s.append(s[k])
        out_l.append(np.full(len(k), c, dtype=int))
    if not out_b:
        return (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))
    boxes = np.concatenate(out_b)
    scores = np.concatenate(out_s)
    labels = np.concatenate(out_l)
    if len(scores) > max_det:
        top = np.argsort(scores)[::-1][:max_det]
        boxes, scores, labels = boxes[top], scores[top], labels[top]
    return boxes, scores, labels


# --------------------------------------------------------------------------
# 后处理：把 [1,84,8400] 解码成原图坐标下的框
# --------------------------------------------------------------------------
_sigmoid_applied: dict[str, bool] = {}


def _needs_sigmoid(cls_scores: np.ndarray) -> bool:
    sample = cls_scores.ravel()[:: max(1, cls_scores.size // 10000)]
    return bool(sample.min() < 0.0 or sample.max() > 1.0)


def decode(pred: np.ndarray, ratio: float, pad, orig_wh,
           conf_thr=0.001, iou_thr=0.7, max_det=300):
    """pred 形状支持 (1,84,8400) 或 (1,8400,84)。"""
    if pred.ndim == 3:
        pred = np.squeeze(pred, 0)
    if pred.shape[0] < pred.shape[1]:      # (84, 8400)
        pred = pred.T                       # -> (8400, 84)
    boxes = pred[:, :4]                     # cx, cy, w, h (在 640 画布上)
    cls_scores = pred[:, 4:]

    if _needs_sigmoid(cls_scores):
        cls_scores = 1.0 / (1.0 + np.exp(-cls_scores))

    labels = cls_scores.argmax(1).astype(int)
    scores = cls_scores.max(1)

    m = scores > conf_thr
    boxes, labels, scores = boxes[m], labels[m], scores[m]
    if boxes.size == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    cx, cy, w, h = boxes.T
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    b = np.stack([x1, y1, x2, y2], 1)

    px, py = pad
    b -= np.array([px, py, px, py], dtype=b.dtype)
    b /= ratio
    ow, oh = orig_wh
    b = np.clip(b, [0, 0, 0, 0], [ow, oh, ow, oh])

    # 过滤退化框
    keep = (b[:, 2] - b[:, 0]) > 1e-3
    keep &= (b[:, 3] - b[:, 1]) > 1e-3
    b, labels, scores = b[keep], labels[keep], scores[keep]
    if b.size == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    return multiclass_nms(b, scores, labels, conf_thr, iou_thr, max_det)


# --------------------------------------------------------------------------
# 数据集：Ultralytics coco128（YOLO 格式标注）
# --------------------------------------------------------------------------
def load_coco128(root: str = "data/coco128"):
    root = Path(root)
    img_dir = root / "images" / "train2017"
    lbl_dir = root / "labels" / "train2017"
    files = sorted(img_dir.glob("*.jpg"))
    if not files:
        sys.exit(f"在 {img_dir} 找不到图片，请先下载 coco128.zip 并解压")

    samples = []
    for f in files:
        im = cv2.imread(str(f))
        if im is None:
            continue
        oh, ow = im.shape[:2]
        gt_b, gt_c = [], []
        lbl = lbl_dir / (f.stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text().strip().splitlines():
                p = line.split()
                if len(p) != 5:
                    continue
                c, cx, cy, bw, bh = (float(v) for v in p)
                x1 = (cx - bw / 2) * ow
                y1 = (cy - bh / 2) * oh
                x2 = (cx + bw / 2) * ow
                y2 = (cy + bh / 2) * oh
                gt_b.append([x1, y1, x2, y2])
                gt_c.append(int(c))
        samples.append({
            "id": f.stem,
            "path": str(f),
            "image": im,
            "gt_boxes": np.asarray(gt_b, dtype=np.float32).reshape(-1, 4),
            "gt_classes": np.asarray(gt_c, dtype=int),
        })
    return samples


# --------------------------------------------------------------------------
# mAP
# --------------------------------------------------------------------------
def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def compute_coco_map(predictions, iou_thrs=np.linspace(0.5, 0.95, 10)):
    """累进式 COCO 风格 mAP。

    predictions: list[dict(gt_boxes, gt_classes, boxes, scores, labels)]
    返回 dict(mAP50_95, mAP50, per_class_ap)
    """
    iou_thrs = iou_thrs
    aps_5095, aps50, detail = [], [], {}

    for cls in range(NUM_CLASSES):
        per_cls = []
        for p in predictions:
            gm = p["gt_classes"] == cls
            dm = p["labels"] == cls
            per_cls.append((p["gt_boxes"][gm], p["boxes"][dm], p["scores"][dm]))
        n_images_with_gt = sum(1 for g, _, _ in per_cls if len(g) > 0)
        npos = sum(len(g) for g, _, _ in per_cls)
        total_dets = sum(len(d) for _, d, _ in per_cls)
        if npos == 0 or total_dets == 0:
            detail[cls] = {"npos": int(npos), "nimg_gt": n_images_with_gt, "imgs": {}, "map": None}
            continue

        # 按分数全局排序
        all_scores, img_ids, det_ids = [], [], []
        for i, (_, d, s) in enumerate(per_cls):
            if len(s) == 0:
                continue
            all_scores.append(s)
            img_ids.append(np.full(len(s), i, dtype=int))
            det_ids.append(np.arange(len(s), dtype=int))
        all_scores = np.concatenate(all_scores)
        img_ids = np.concatenate(img_ids)
        det_ids = np.concatenate(det_ids)
        order = np.argsort(-all_scores)
        all_scores, img_ids, det_ids = all_scores[order], img_ids[order], det_ids[order]

        # 预计算每个图的 IoU
        iou_cache = {}
        for i, (g, d, _) in enumerate(per_cls):
            if len(g) and len(d):
                iou_cache[i] = _iou_matrix(d, g)

        ap_per_iou = []
        for thr in iou_thrs:
            matched = {i: np.zeros(len(per_cls[i][0]), dtype=bool) for i in range(len(per_cls))}
            tp = np.zeros(len(all_scores))
            fp = np.zeros(len(all_scores))
            for k in range(len(all_scores)):
                i, d_i = img_ids[k], det_ids[k]
                g = per_cls[i][0]
                if len(g) == 0:
                    fp[k] = 1
                    continue
                ious = iou_cache[i][d_i]
                cand = np.where((ious >= thr) & (~matched[i]))[0]
                if cand.size:
                    j = cand[np.argmax(ious[cand])]
                    matched[i][j] = True
                    tp[k] = 1
                else:
                    fp[k] = 1
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            rec = cum_tp / max(npos, 1e-9)
            prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
            # 101 点插值
            pts = np.linspace(0, 1, 101)
            ap = float(np.mean([prec[rec >= t].max() if np.any(rec >= t) else 0.0 for t in pts]))
            ap_per_iou.append(ap)

        m5095 = float(np.mean(ap_per_iou))
        aps_5095.append(m5095)
        aps50.append(ap_per_iou[0])
        detail[cls] = {
            "npos": int(npos),
            "nimg_gt": n_images_with_gt,
            "map50_95": m5095,
            "imgs": {},
        }

    return {
        "mAP50-95": float(np.mean(aps_5095)) if aps_5095 else float("nan"),
        "mAP50": float(np.mean(aps50)) if aps50 else float("nan"),
        "classes_evaluated": len(aps_5095),
        "per_class": detail,
    }
