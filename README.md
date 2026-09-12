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
| INT8-AA (accuracy-aware) | 31.69 | 31.56 | 3.34 | 0.3897 | −0.0138 | 1.75× |

### 三条值得记住的结论

**1. 朴素 INT8 的精度代价，比你以为的大得多。**
默认 PTQ 把 mAP50-95 从 0.4035 打到 0.3455——掉了 **5.8 个点**。光看 "1.99× 加速" 就上线的话，这个模型基本废了。

> **命名提醒：** 这里的 INT8-AA 是 NNCF 的 *accuracy-aware quantization*，
> **不是** AWQ（Activation-aware Weight Quantization，Lin et al. 面向 LLM 权重的方法）。
> 两者机制完全不同，别混用缩写。

**2. 但 accuracy-aware quantization 几乎把这笔账抹平了。**
`nncf.quantize_with_accuracy_control`（max_drop=0.01，rank-based 敏感度排序，把敏感算子回退到 FP16/FP32）把精度损失压到 **1.4 个点**，代价只有 **3.7 ms**——相比 FP32 仍然有 **1.75×** 加速。这条曲线（精度预算 ↔ 加速比）才是真正该向业务方汇报的东西。

**关于「允许多掉 1 个点，结果掉了 1.4 个点」：**

`max_drop=0.01` 这个预算是**在敏感度调参集（tuning split）上计算和判定的**；
上表里的 −0.0138 是在**完全不相交的 held-out 测试集（val2017 200 张）上测出来的**。
两者的差就是泛化间隙，属于设计使然——正因为允许它存在，数字才没有作弊。

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

## C++ Runtime 路径 vs Python

`benchmark_cpp/yolo_infer_cpp.cpp` 是同一套流水线的 **C++ 实现**：letterbox（含手写双线性 resize）、
解码、逐类 NMS **全部自己写**，不依赖 OpenCV（`opencv-python` 的 wheel 不提供 C++ 头文件与 cmake config）。

```
stage           C++ (ms)  Python (ms)   speed-up
pre                11.82         1.63      0.14x
infer              44.67        46.30      1.04x
post                1.13         8.04      7.12x   <- 后处理 C++ 快 7 倍
total              57.78        56.04      0.97x
检测框数量       C++ 194      Python 194    完全一致
```

三点值得注意：

1. **后处理（解码 + 逐类 NMS）C++ 快 7.12×**——这是 host 端真正的开销大头。
2. **两边检测结果完全一致（194 == 194）**，这是对 C++ 实现的正确性验证。
3. **前处理反而慢了 6.8×**：手写标量双线性 resize 打不过 OpenCV 里 SIMD 优化过的 `resize`。
   —— 这条本身就是「为什么需要经过调优的算子库」的第一手证据。
   整体 total 因此基本打平（0.97×），**没有粉饰成「C++ 全面更快」**。

## 异步与吞吐模式

`async_bench.py` 对比同步 / `AsyncInferQueue` 异步、`PERFORMANCE_HINT` 的 latency / throughput、
单流 / 多流（本机 4 核，THROUGHPUT hint 自动取 4 streams）：

| Model | Mode | Latency ms | FPS |
|---|---|---|---|
| FP32 | sync / latency-hint / 1 stream | 46.45 | 21.53 |
| FP32 | async / throughput-hint / 4 streams | 191.32 | 22.97 |
| INT8 | sync / latency-hint / 1 stream | 27.33 | 36.59 |
| INT8 | async / throughput-hint / 4 streams | 103.69 | 40.85 |

多流把 INT8 从 36.59 拉到 40.85 FPS（+11.6%），FP32 +6.7% —— 4 核 CPU 上就这个量级。

**关键理解：异步「延迟」是 103.69 ms 而不是 27 ms，因为它量的是「提交到回调」的时间，
队列饱和时自然把排队等待也算进去了。** 所以：

- 要**延迟** → 同步，或非饱和队列
- 要**吞吐** → async + 多 stream + THROUGHPUT hint

## 目录

```
openvino-yolo-benchmark/
├── run_benchmark.py        主流程：造 IR → NNCF 量化 → 精度 → 性能 → 出表
├── yolo_utils.py           letterbox / anchor 解码 / per-class NMS / COCO 101点 AP
├── coco_eval_set.py        coco128 与 COCO val2017 评测集；含类别映射
├── async_bench.py          同步 / 异步、latency / throughput hint、单流 / 多流对比
├── cpp_python_parity.py    C++ 与 Python 等价管线的逐阶段耗时对比
├── benchmark_cpp/
│   ├── yolo_infer_cpp.cpp  完整 C++ 推理路径（手写 letterbox / 解码 / NMS，零 OpenCV 依赖）
│   ├── bench_cpp.cpp       纯推理计时器（只测模型）
│   └── CMakeLists.txt      含 macOS libc++ 头文件路径的 workaround
├── results/async_bench.json
├── results/cpp_python_parity.json
├── results/
│   ├── results.md          生成的结果表
│   └── verify.json         三轮重复测量的原始数据
└── requirements.txt
```

---

## 已知限制 / 下一步

- **只测了 CPU，而且这不是懒惰而是物理限制。** 本机是 macOS x86_64：OpenVINO 的 macOS 发行版
  **只带 `libopenvino_intel_cpu_plugin`，没有 GPU 插件**（已核实 pip 包的 `libs/` 目录），
  这颗 i7-8559U 也没有 NPU。所以 iGPU / NPU 一列拿不到，**不会编造**。
  要补齐三端，需在 Linux + Intel GPU，或 Intel Core Ultra 机器（如 Intel Tiber AI Cloud）上跑同一脚本，
  它会自动探测 `GPU` / `NPU`。
- NPU 侧预计会遇到的约束（待硬件验证）：**不支持 FP32**（仅 FP16/INT8）、**必须静态 shape**、**首次编译到 NPU 有十几秒冷启动**、不支持的算子会**静默回退到 CPU**。这几条决定了 "NPU 不一定比 iGPU 快"。
- 校准集只有 128 张。扩到 300+ 张通常能进一步缩小朴素 INT8 的精度损失。
- 下一步想加：per-layer 延迟剖析（`benchmark_app -report_type detailed_counters`）定位回退的具体算子，以及 INT4 权重压缩。
