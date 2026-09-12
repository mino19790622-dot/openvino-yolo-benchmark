"""评测集封装：支持 Ultralytics coco128 与 COCO val2017 官方子集。

坑提醒：COCO 的 category_id 是 1..90 且不连续（person=1, bicycle=2, ... toothbrush=90），
而 YOLO 输出的是 0..79 的连续索引。必须建立「按 category_id 升序排列」的映射，
这是 YOLO/COCO 的通用约定，弄错这一步 mAP 会直接崩到接近 0。
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data"
VAL_IMG_DIR = DATA / "val2017_images"
VAL_ANN = DATA / "annotations" / "instances_val2017.json"
VAL_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"

import yolo_utils as U


class Coco128Set:
    """128 张图，来自 COCO train2017 子集。注意：模型训练时见过这些数据，
    绝对 mAP 会偏乐观，不适合单独引用；用同一套数据比较 FP32 vs INT8 的差值仍然有效。"""

    def __init__(self):
        self.samples = U.load_coco128(str(DATA / "coco128"))

    def name(self) -> str:
        return "coco128 (subset of COCO train2017)"

    def n(self) -> int:
        return len(self.samples)

    def n_gt(self) -> int:
        return int(sum(len(s["gt_boxes"]) for s in self.samples))

    def images(self):
        for s in self.samples:
            yield s["image"], {"gt_boxes": s["gt_boxes"], "gt_classes": s["gt_classes"]}


class Val2017Set:
    """COCO 官方 val2017 子集 —— 模型没见过的数据，绝对 mAP 可引用。"""

    def __init__(self, limit: int = 300):
        if not VAL_ANN.exists():
            raise FileNotFoundError(
                f"缺少 {VAL_ANN}\n"
                "下载地址: http://images.cocodataset.org/annotations/annotations_trainval2017.zip\n"
                "解压后把 annotations/instances_val2017.json 放到 data/annotations/"
            )
        ann = json.loads(VAL_ANN.read_text())

        cat_ids = sorted(c["id"] for c in ann["categories"])
        cat2idx = {c: i for i, c in enumerate(cat_ids)}
        self.n_classes = len(cat_ids)

        img_info = {im["id"]: im for im in ann["images"]}
        boxes_by_img: dict[int, list] = {i: [] for i in img_info}
        for a in ann["annotations"]:
            if a.get("iscrowd", 0):
                continue                     # 跳过 crowd，与 COCO 官方评测一致
            x, y, w, h = a["bbox"]           # xywh，左上角在原图坐标系
            boxes_by_img.setdefault(a["image_id"], []).append(
                ([x, y, x + w, y + h], cat2idx[a["category_id"]])
            )

        ids_with_gt = [i for i in boxes_by_img if len(boxes_by_img[i]) > 0]
        ids_with_gt.sort()
        self.ids = ids_with_gt[:limit]
        self.img_info = img_info
        self.boxes_by_img = boxes_by_img
        VAL_IMG_DIR.mkdir(parents=True, exist_ok=True)

    def name(self) -> str:
        return f"COCO val2017 subset ({len(self.ids)} images)"

    def n(self) -> int:
        return len(self.ids)

    def n_gt(self) -> int:
        return int(sum(len(self.boxes_by_img[i]) for i in self.ids))

    def _fetch(self, img_id: int) -> Path:
        p = VAL_IMG_DIR / f"{img_id:012d}.jpg"
        if not p.exists():
            urllib.request.urlretrieve(VAL_URL.format(img_id), p)
        return p

    def images(self):
        for idx in self.ids:
            p = self._fetch(idx)
            im = cv2.imread(str(p))
            if im is None:
                continue
            pairs = self.boxes_by_img[idx]
            gt = np.asarray([b for b, _ in pairs], dtype=np.float32).reshape(-1, 4)
            gc = np.asarray([c for _, c in pairs], dtype=int)
            yield im, {"gt_boxes": gt, "gt_classes": gc}


def make_eval_set(which: str = "coco128", **kw):
    """工厂函数：按名字构造评测集。

    which='coco128'  → 128 张，本地已有，秒开
    which='val2017'  → COCO 官方验证子集，需要联网下载图片
    """
    if which == "val2017":
        return Val2017Set(**kw)
    return Coco128Set()
