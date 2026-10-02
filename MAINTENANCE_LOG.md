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

## 2026-09-30 — yolo_utils.py: 补返回类型标注 + 清掉 5 处既有 ruff 告警

- **Change**: yolo_utils.py: (1) 给 letterbox/preprocess/multiclass_nms/decode/load_coco128/compute_coco_map 补返回类型标注；为 preprocess、load_coco128（此前无 docstring）、decode 补/扩 docstring。 (2) 清掉该文件里 5 处【既有】ruff 告警 —— verify 只 lint 本次改动的文件，不清干净就无法通过：F401 删掉未使用的 import os（全仓 grep 无 yolo_utils.os / U.os 引用）；RUF046 x2 去掉 int(round(x)) 多余的 int()（Python 3 的 round() 已返回 int）；B008 把默认参数里的 np.linspace(0.5,0.95,10) 提为模块级常量 IOU_THRS；PLW0127 删掉空操作行 iou_thrs = iou_thrs。 全部为行为等价改动：未改业务逻辑、未增删依赖、未动 .github/workflows。
- **Verification**: pytest: 22 passed; ruff check yolo_utils.py: All checks passed; 另做 20 万组随机尺寸的 round() vs int(round()) 等价性复算，差异 0

## 2026-10-01 — docs(utils): document the two private helpers

- **Change**: 给 yolo_utils.py 的 _needs_sigmoid 与 _iou_matrix 补 docstring：说明前者用抽样 min/max 区分 logits 与概率，后者返回 (len(a), len(b)) 的成对 IoU 矩阵并对空输入与退化框兜底。纯文档，无行为变化。
- **Verification**: pytest 22 passed; ruff All checks passed

## 2026-10-02 — docs(eval): document the two eval-set constructors and name()

- **Change**: coco_eval_set.py: add docstrings to Coco128Set.__init__ (reads all of data/coco128 into memory up front) and Coco128Set.name()/Val2017Set.name() (human-readable label that is printed and written into the report header), plus Val2017Set.__init__ (keeps only images with >=1 non-crowd box, sorted by id, capped at limit; categories remapped ascending from category_id to 0..79). Pure documentation, no behaviour change.
- **Verification**: pytest: 22 passed; ruff check coco_eval_set.py: All checks passed

