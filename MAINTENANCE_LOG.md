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

## 2026-10-05 — test(iou): cover empty and zero-area boxes in _iou_matrix

- **Change**: tests/test_yolo_utils.py: 给 _iou_matrix 补 2 个测试 —— ① 任一侧为空时返回形状正确的全零矩阵 ((1,0)/(0,1)/(0,0))，验证调用方无需先做空判断；② 零面积退化框使并集为 0，靠 1e-9 兜底返回 0.0 而非 nan/inf（np.isfinite 断言），保证 mAP 不会被污染。纯测试新增，无生产代码改动。
- **Verification**: pytest 24 passed (24 collected in tests/test_yolo_utils.py, 由 22 -> 24); ruff check tests/test_yolo_utils.py: All checks passed

## 2026-10-06 — Add edge-case tests for NMS and decode

- **Change**: tests/test_yolo_utils.py: +4 tests covering previously untested branches - single-box NMS (order.size==1 early break), multiclass_nms on empty input (the 'if not out_b' early return), decode's zero-area degeneracy guard (w=h=0 passes conf_thr but must be dropped), and a 2x2 px non-degenerate box that must survive the same guard. Test-only, production code untouched.
- **Verification**: pytest 28 passed (was 24); ruff All checks passed

## 2026-10-07 — multiclass_nms 的 conf_thr 是空参数

- **Change**: yolo_utils.py 的 multiclass_nms() 收下 conf_thr 但函数体从未引用它（grep 全仓：该名字只在签名第 74 行与调用方 decode() 第 156 行出现）；真正按置信度过滤的是 decode() 里的 scores > conf_thr。补 docstring 写明这一点，避免有人直接调用 multiclass_nms 时误以为传 conf_thr 就能裁剪。同时写明 max_det 是在各类别结果拼接之后全局施加（保留全局 top 分），以及空输入返回 (0,4)/(0,)/(0,) 三件套。仅文档，无行为变更。
- **Verification**: pytest 28 passed（与 --collect-only 的 tests/test_yolo_utils.py: 28 一致）；ruff check yolo_utils.py -> All checks passed

## 2026-10-08 — docs(eval): make_eval_set 的 **kw 只转发给 Val2017Set

- **Change**: 给 make_eval_set 补 docstring：**kw 只转发给 Val2017Set；Coco128Set 不接受任何参数，故 make_eval_set('coco128', limit=50) 里的 limit 会被静默丢弃（不报错也不裁剪，仍跑满 128 张）。run_benchmark.py --limit 同理，仅在 --eval-set val2017 时生效。纯文档，无行为变化。
- **Verification**: pytest 28 passed；ruff 仅 lint 改动文件 coco_eval_set.py → All checks passed

## 2026-10-09 — test(map): cover GT-only and GT-free images in compute_coco_map

- **Change**: Add 2 tests to tests/test_yolo_utils.py. (1) an image that has ground truth but zero detections is skipped while ranking yet still counted in npos -> mAP50 = 51/101. (2) a detection on an image with no ground truth can never match, so when it outranks the single true positive precision at recall 1.0 drops to 1/2 -> mAP50 = 0.5. Both branches were previously untested (coverage 84% -> 86%). Assertion values were obtained by running compute_coco_map first, not estimated. No production code touched.
- **Verification**: ok=true; pytest 30 passed (was 28, +2); ruff All checks passed on tests/test_yolo_utils.py

## 2026-10-15 — docs(parity): document the C++/Python host-side contract

- **Change**: cpp_python_parity.py: 给 dump_raw_bgr / run_cpp / run_python 三个此前无 docstring 的函数补上文档，记录两侧接口契约 —— (1) dump_raw_bgr 写出的裸格式是 int32 H + int32 W + H*W*3 字节 BGR，与 yolo_infer_cpp.cpp 声明的 raw_bgr.bin 格式一一对应，任一侧改头字段顺序 parity 即失去意义；(2) run_cpp 故意只传 4 个参数不传 conf/iou，依赖 C++ 侧默认 conf=0.001 / iou=0.7 与 U.decode() 默认参数相同，故两边仍是同一套阈值；(3) run_python 丢弃前 5 次 warm-up 后取各阶段中位数（与 C++ 侧 warmup=5 口径一致），dets 取最后一次迭代的框数用于与 C++ 核对。同时清掉该文件 5 处【既有】ruff 告警：EXE001 给带 shebang 的脚本 chmod +x；PLW1510 显式 check=False；RUF059 x2 未使用的 s/l 改名为 _s/_l；F541 去掉无占位符的 f 前缀。全部为行为等价改动：未改业务逻辑、未重构、未增删依赖、未动 .github/workflows。
- **Verification**: ok=true; pytest 30 passed; ruff check cpp_python_parity.py: All checks passed; py_compile 通过

## 2026-10-16 — docs(async): document build_input/measure_sync; clear 2 lint errors

- **Change**: async_bench.py: 给两个此前无 docstring 的函数补文档 —— (1) build_input 用 default_rng(0) 造确定性输入，形状取自 compiled.input().get_shape()，故 FP32/INT8 两模型输入签名一致时拿到的是逐元素相同的数据，延迟差才是纯量化差异；固定种子保证跨轮次可复现（换数据会改缓存命中模式，数字会漂）。(2) measure_sync 写明统计口径：前 10 次 warmup 不计入（首次 infer 含插件初始化与缓存建立）；计时只包住 req.infer()，set_input_tensor 的拷贝开销被排除；取中位数而非均值以抗调度抖动；并点明其 fps = 1000/中位延迟 是延迟的倒数，与 measure_async 的 fps 不是同一个量。同时清掉该文件 2 处【既有】ruff 告警：EXE001 给带 shebang 的脚本 chmod +x（git 文件模式 100644->100755，唯一修法）；PLC0206 把 for tag in results: / results[tag].items() 改为 for tag, per_model in results.items(): / per_model.items()（仅最后一处 markdown 表格生成循环，已用等价性脚本验证新旧两版输出逐行相同）。全部行为等价：未改业务逻辑、未重构、未增删依赖、未动 .github/workflows。
- **Verification**: ok=true; pytest 30 passed (与 --collect-only 的 tests/test_yolo_utils.py: 30 自洽); ruff check async_bench.py: All checks passed; py_compile 通过; 新旧循环输出等价性脚本 identical=True

## 2026-10-19 — document the measure_async timing and throughput contract

- **Change**: async_bench.py: 给 measure_async 补 docstring，记录三条此前只存在于代码里的口径 —— ① warmup 提交同样会进回调，靠 wait_all() 后 latencies.clear() 丢弃；② 单请求延迟以该请求的提交时刻（经 userdata 传入）为起点，因此是含排队等待的队列驻留时间，深队列下必然大于 measure_sync 的同类数字；③ fps 用 iters*1000/wall 算，是聚合吞吐，与 measure_sync 里中位延迟取倒数的 fps 不是同一个量。另记录返回三键及无回调时 latency_ms 为 nan、wall_ms 仍有效。
- **Verification**: pytest: 30 passed（collect-only 复算 files=1 total_tests=30）；ruff check async_bench.py: All checks passed

## 2026-10-20 — docs(map): 修正 compute_coco_map 返回值文档并写明跳过语义

- **Change**: yolo_utils.py::compute_coco_map() 的 docstring 与代码实际返回不符：原文写 'mAP50_95' 与 'per_class_ap'，实测两键均不存在（KeyError），真实键为 'mAP50-95'（连字符）/ 'mAP50' / 'classes_evaluated' / 'per_class'。改写后写明五条契约，全部经探针实测：① 顶层仅四键，mAP 键用连字符；② per_class 恒含 80 个 int 键，但条目键名不一致——计分条目有 map50_95，被跳过条目只有 map=None；③ imgs 恒为 {}（预留未填充）；④ 跳过语义：npos==0 或 total_dets==0 的类别不进均值而非记 0 分，故 mAP 分母是 classes_evaluated 不是 80，整类漏检不会被惩罚，指标偏乐观；⑤ 无类别计分时 mAP 为 nan 而非 0。+25/-3，纯文档。
- **Verification**: pytest 30 passed; ruff All checks passed (yolo_utils.py)

## 2026-10-21 — docs(readme): 修正目录树重复节点并补上缺失的 tests/

- **Change**: README.md 的目录树有三处与 git ls-files 不符：① results/ 节点重复出现两次（一次只列两个 json、一次列 results.md 与 verify.json），树结构本身是坏的；② 漏了 results/results.json；③ tests/conftest.py、tests/test_yolo_utils.py、pytest.ini 整块缺失，README 全文从未提到仓库有测试。按 git ls-files 逐条重排目录树，并补一段说明：yolo_utils.py 只 import numpy 与 OpenCV，故不需要模型文件、也不 import openvino 就能跑测试。另把已跟踪但未列出的 benchmark_cpp/yolo_infer 标为早期草稿（实测 98 行 vs yolo_infer_cpp.cpp 308 行，文件头同名、措辞更旧）。纯文档 +16/-3，无代码改动。
- **Verification**: ok=true; pytest 30 passed; 仅改 .md 故 lint 步骤跳过; git diff --stat 显示 README.md 1 file changed, 16 insertions(+), 3 deletions(-)

## 2026-10-22 — docs(parity): pin the iters>=6 precondition of run_python

- **Change**: cpp_python_parity.py::run_python 补 docstring，+9 行，纯文档。写明 warm-up=5 带来的隐含下限：iters<=5 时四个采样列表全空、statistics.median([]) 抛 StatisticsError，而 --iters 在 main() 里无下限校验，故错发生在 C++ 侧已跑完之后；同一边界上两侧不一致——C++ median_of() 对空 vector 返回 0.0（打印 0.00 ms，像极快而非无数据），Python 抛异常；dets 只在 it>=5 分支赋值故退化输入下停在 0，而 C++ last_dets 每次迭代都更新，该格不可比。三条均用一次性探针实测（iters=4/5 -> samples=0 且 EXC StatisticsError；iters=6 -> samples=1）并核对 C++ 源码 median_of/yolo_infer_cpp.cpp:218-228,268-292 后落笔。
- **Verification**: ok:true — pytest 30 passed；ruff All checks passed（仅 lint 改动文件 cpp_python_parity.py）

## 2026-10-24 — docs(async): document the AsyncInferQueue callback contract

- **Change**: 为 async_bench.py 的 measure_async 内部回调 cb() 补 docstring：钉住三件事 —— userdata 必须是提交时刻的 perf_counter()（漏传会在回调线程抛 TypeError）、回调跑在 OpenVINO 工作线程故不加锁但落库顺序不等于完成顺序（只能取中位数）、预热样本由调用方 wait_all() 后 latencies.clear() 统一丢弃。纯注释改动，不改运行时行为。
- **Verification**: pytest 30 passed（30 个点、无 F/E）；ruff check async_bench.py: All checks passed

