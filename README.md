# openvino-yolo-benchmark

把 YOLOv8n 从 FP32 一路压到 INT8，用**可复现的实测数字**回答一个问题：
**量化到底买到了多少性能、付出了多少精度？**

全部数字由 `run_benchmark.py` 在本机真实跑出来，无任何臆测值。

---

## 结果一览

**硬件**：MacBook Pro 2018 · Intel Core i7-8559U @ 2.70 GHz（Coffee Lake，4C/8T）
**软件**：OpenVINO 2025.4.1 · NNCF 3.2.0 · Python 3.13 · macOS 15.7 (x86_64)
**评测集**：COCO val2017 子集，200 张图 / 1444 个 ground-truth 框
**方法**：同步推理，每轮 200 次迭代 + 10 次 warm-up，取中位数；**重复 3 轮，再取中位数的中位数**

| Precision | Latency (ms) | Throughput (FPS) | Weights (MB) | mAP50-95 | Δ mAP50-95 | Speed-up |
|---|---|---|---|---|---|---|
| FP32 | 55.60 | 17.99 | 12.71 | 0.4035 | — | 1.00× |
| FP16 | 53.27 | 18.77 | 6.35 | 0.4030 | −0.0005 | 1.04× |
| INT8 (PTQ) | 27.99 | 35.73 | 3.29 | 0.3455 | −0.0580 | **1.99×** |
| INT8 (accuracy-aware) | 31.69 | 31.56 | 3.34 | 0.3897 | −0.0138 | 1.75× |

### 三条值得记住的结论

**1. 朴素 INT8 的精度代价，比你以为的大得多。**
默认 PTQ 把 mAP50-95 从 0.4035 打到 0.3455——掉了 **5.8 个点**。光看 "1.99× 加速" 就上线的话，这个模型基本废了。

**2. 但 accuracy-aware quantization 几乎把这笔账抹平了。**
`nncf.quantize_with_accuracy_control`（max_drop=0.01，rank-based 敏感度排序，把敏感算子回退到 FP16/FP32）把精度损失压到 **1.4 个点**，代价只有 **3.7 ms**——相比 FP32 仍然有 **1.75×** 加速。这条曲线（精度预算 ↔ 加速比）才是真正该向业务方汇报的东西。

**3. FP16 在这颗 CPU 上几乎不加速，原因很有意思。**
i7-8559U 是 Coffee Lake（第 8 代），**没有原生 FP16 计算通路**，FP16 在 CPU 上要上转成 FP32 再算。所以体积虽然减半（12.71 → 6.35 MB），延迟只从 55.60 降到 53.27 ms（1.04×），收益基本来自内存带宽。

> **这正是 NPU 存在的理由。** 同一份 INT8 模型，在这颗没有 VNNI 指令的 CPU 上要靠 AVX2 模拟摆弄，跑到 27.99 ms；放到带 INT8 矩阵单元的专用 NPU 上，数量级是完全不同的。本项目把「通用 CPU 的瓶颈」量化了出来——这对判断一个问题值不值得下沉到硬件，是必要的输入。

---

## 为什么这个仓库不用 ultralytics / PyTorch

本机是 **macOS x86_64**，而：

- PyTorch 官方**不为该平台提供 wheel**（只提供 arm64），`pip install torch` 直接 no matching distribution
- OpenVINO 在 macOS 上**只注册到 CPU**（无 GPU 插件），且这颗芯片没有 NPU

所以整条链路改用官方 `yolov8n.onnx` + `ovc` + NNCF + OpenVINO Runtime，解码/NMS/mAP 全部用 numpy 手写。副作用是：**这个仓库在任何机器上都能跑，不需要装 PyTorch。**

---

## 快速开始

```bash
git clone https://github.com/mino19790622-dot/openvino-yolo-benchmark
cd openvino-yolo-benchmark

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 全量：量化 + 精度评估 + 性能基准
python run_benchmark.py --eval-set val2017 --limit 200 --accuracy-control

# 只想快速看性能
python run_benchmark.py --skip-accuracy
```

脚本会自动下载 `yolov8n.onnx` 与 `coco128`；使用 `--eval-set val2017` 时会自动取 COCO 官方验证子集。

### 版本组合是个坑

`nncf==3.3.0` 依赖 `ov.Type.u2`，而 OpenVINO 当前最新版 `2025.4.1` **并未暴露该枚举**，`import nncf` 阶段就会 `AttributeError`。
**必须用 `nncf==3.2.0`**，已在 `requirements.txt` 里锁死。

---

## 方法学（面试官会问的三件事）

**1. 延迟还是吞吐？**
这里报的是**单次同步推理延迟**（等价于 `benchmark_app -api sync -nstreams 1`），batch size = 1，静态 shape 640×640。没有做并发填充，所以不是吞吐峰值。

**2. 测的是模型还是整条 pipeline？**
只测模型推理，**不含** letterbox 前处理和 NMS 后处理。在本机这颗 CPU 上整 pipeline 会比这些数字再慢一截。

**3. 为什么要跑 3 轮？**
笔记本有热节流，单次测量不可信。实测同一模型三次的中位数最大波动达到 **8.36 ms**（首轮偏慢，是频率爬升 + 首次编译的冷启动效应）。所以每个格子是 **3 轮 × 200 次迭代，取中位数的中位数**，圆整后 round-to-round spread ≤ 1.7 ms。

**4. 数据集是分开的。**
- **校准集**：coco128 后 96 张（做统计量收集）
- **敏感度调参集**：coco128 前 32 张（accuracy-aware 回退决策用）
- **测试集**：COCO val2017 200 张（整个量化流程都没碰过）

三份严格不重叠。这点很重要——否则 accuracy-aware 就是在测试集上调参，报出来的数字没有意义。

---

## 校验：这套 mAP 实现可信吗？

自己手写的解码 + NMS + COCO AP，怎么证明没写错？拿官方数字对。

| | 本项目（val2017 子集 200 张） | Ultralytics 官方（val2017 全量 5000 张） |
|---|---|---|
| mAP50-95 | **0.4035** | 0.370 |
| mAP50 | **0.5543** | 0.539 |

mAP50 只差 **+0.015**，完全落在 200 张子集的抽样波动范围内（本子集按 image id 取前 200 张，内容分布天然与全量有偏差）。
→ 说明 letterbox 预处理、8400 anchor 解码、坐标还原、per-class NMS、以及**关键的那一步**——COCO `category_id`（1…90，不连续）到 YOLO 连续索引（0…79）的按序映射——全部正确。

> 踩坑记录：category 映射弄错的话，mAP 会直接崩到接近 0，而且报出来的框看着"还挺像回事"，很容易骗过肉眼检查。

**另一个坑**：`coco128` 采自 COCO **train2017**，模型训练时见过这批图，在它上面得到 **mAP50-95 = 0.4444**，明显高于 val2017 的 0.4035。所以**绝对 mAP 必须在模型没见过的数据上报告**。仓库里保留了 `Coco128Set` 只用于校准与快速冒烟。

---

## 目录

```
openvino-yolo-benchmark/
├── run_benchmark.py        主流程：造 IR → NNCF 量化 → 精度 → 性能 → 出表
├── yolo_utils.py           letterbox / anchor 解码 / per-class NMS / COCO 101点 AP
├── coco_eval_set.py        coco128 与 COCO val2017 评测集；含类别映射
├── benchmark_cpp/          OpenVINO C++ Runtime 基准（测 C++ vs Python 端到端开销）
├── results/
│   ├── results.md          生成的结果表
│   └── verify.json         三轮重复测量的原始数据
└── requirements.txt
```

---

## 已知限制 / 下一步

- **只测了 CPU。** 这台机器没有可用的 Intel GPU / NPU 设备。要在 Core Ultra 上补齐三端对比，直接运行同一脚本即可（`--eval-set val2017`），脚本会自动探测 GPU / NPU。
- NPU 侧预计会遇到的约束（待硬件验证）：**不支持 FP32**（仅 FP16/INT8）、**必须静态 shape**、**首次编译到 NPU 有十几秒冷启动**、不支持的算子会**静默回退到 CPU**。这几条决定了 "NPU 不一定比 iGPU 快"。
- 校准集只有 128 张。扩到 300+ 张通常能进一步缩小朴素 INT8 的精度损失。
- 下一步想加：per-layer 延迟剖析（`benchmark_app -report_type detailed_counters`）定位回退的具体算子，以及 INT4 权重压缩。
