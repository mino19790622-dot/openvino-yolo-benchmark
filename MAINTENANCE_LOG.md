# Maintenance log

> This file is appended automatically by a scheduled maintenance task, and the
> commits that touch it carry `[skip ci]`. Every entry corresponds to one real
> change that passed the repository's own checks before it was pushed.
>
> The entries are generated rather than hand-written, and they record
> maintenance work only — they are not a measure of manual development effort.

## 2026-09-22 — add a first unit-test suite for the torch-free pipeline maths

- **Change**: new tests/ covering letterbox, preprocess, nms, multiclass_nms, decode and the IoU/sigmoid helpers (16 tests); adds pytest.ini and a conftest.py that puts the repo root on sys.path
- **Verification**: pytest: 16 passed; ruff: all checks passed

## 2026-09-23 — note that Val2017Set.images() skips unreadable images

- **Change**: coco_eval_set.py: add a docstring to Val2017Set.images() recording that images are downloaded lazily and unreadable ones are skipped silently, so the number yielded can be lower than n() while n_gt() still counts every id
- **Verification**: pytest: 16 passed; ruff: all checks passed

## 2026-09-24 — document the n()/n_gt()/images() counting contract on both eval sets

- **Change**: coco_eval_set.py: add docstrings to Coco128Set.n()/n_gt()/images() and Val2017Set.n()/n_gt()/_fetch(). Records that n() is the planned image count while images() may yield fewer for Val2017Set (lazy download, unreadable files skipped) but always equals n() for Coco128Set, which reads every image up front.
- **Verification**: pytest: 16 passed; ruff: all checks passed

## 2026-09-25 — 为 compute_coco_map 补 6 个测试（此前零覆盖）

- **Change**: 给 tests/test_yolo_utils.py 增加 mAP 一节，覆盖 yolo_utils.compute_coco_map：完美检测 mAP50=mAP50-95=1；纯误检 mAP=0；10 个 COCO IoU 阈值取平均（IoU=0.8 → mAP50 保持 1.0 而 mAP50-95=0.7）；101 点插值（2 GT 1 检测 → AP=51/101）；两个类别各自计分（classes_evaluated=2）；无任何检测时 classes_evaluated=0 且 mAP50 为 nan。断言值均先在 venv312 里实跑取得，非估算。
- **Verification**: pytest tests: 22 passed（改前 16，本次 +6，collect-only 复算 files=1 total_tests=22）；ruff check tests/test_yolo_utils.py: All checks passed

