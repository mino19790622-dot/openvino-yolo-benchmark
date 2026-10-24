#!/usr/bin/env python3
"""OpenVINO runtime 的同步 / 异步、latency / throughput hint、单流 / 多流对比。

补上 runtime 组日常在聊的三个概念，并给出本机实测数字：
    · synchronous vs asynchronous (AsyncInferQueue)
    · PERFORMANCE_HINT = LATENCY vs THROUGHPUT
    · 单 stream vs 多 stream 聚合吞吐

用法：
    .venv/bin/python async_bench.py [--iters 200]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import openvino as ov

MODELS = {
    "FP32": "models/yolov8n_fp32/yolov8n_fp32.xml",
    "INT8": "models/yolov8n_int8/yolov8n_int8.xml",
}


def build_input(compiled) -> np.ndarray:
    """按模型声明的输入形状造一份**确定性**的随机输入。

    形状取自 ``compiled.input().get_shape()``，所以 FP32 与 INT8 两个模型只要输入签名
    一致，拿到的就是逐元素相同的数据 —— 两者之间的延迟差才是纯粹的量化差异，
    而不是「喂了不同数据」。随机数种子固定为 0（``default_rng(0)``），
    保证跨轮次、跨进程可复现：换一批数据会换掉缓存命中模式，benchmark 数字就漂了。
    """
    shape = list(compiled.input().get_shape())
    rng = np.random.default_rng(0)
    return rng.random(shape, dtype=np.float32)


def measure_sync(compiled, data, iters: int, warmup: int = 10) -> dict:
    """单请求串行推理，返回中位延迟与由其倒数推出的 FPS。

    口径说明：
        · 前 ``warmup`` 次不计入统计 —— 首次 infer 含插件初始化与缓存建立，明显偏慢；
        · 计时只包住 ``req.infer()``，``set_input_tensor`` 的拷贝开销被排除在外；
        · 取**中位数**而非均值，避免偶发调度抖动把结果拉高。

    返回 ``{"latency_ms", "fps"}``，其中 fps 是单流串行的倒数（1000 / 中位延迟），
    所以它衡量的是**延迟**而非吞吐，与 :func:`measure_async` 的 fps 不是同一个量。
    """
    req = compiled.create_infer_request()
    for _ in range(warmup):
        req.set_input_tensor(ov.Tensor(data))
        req.infer()
    samples = []
    for _ in range(iters):
        req.set_input_tensor(ov.Tensor(data))
        t0 = time.perf_counter()
        req.infer()
        samples.append((time.perf_counter() - t0) * 1000)
    med = statistics.median(samples)
    return {"latency_ms": round(med, 2), "fps": round(1000 / med, 2)}


def measure_async(compiled, data, iters: int, nstreams: int, warmup: int = 10) -> dict:
    """用一个深度为 nstreams 的 AsyncInferQueue 提交 iters 次，测总墙钟吞吐与单请求延迟。

    口径说明：
        · 前 ``warmup`` 次提交不计入统计 —— 回调同样会 append，故 ``wait_all()`` 之后
          直接 ``latencies.clear()`` 把它们丢掉，而不是另开一个计数器；
        · 单请求延迟由回调测得，起点是该请求的**提交时刻**（经 ``userdata`` 传入），
          终点是完成时刻 ⇒ 它包含排队等待，是**队列驻留时间**而非纯计算时间；
          队列越深，这个数就越大于 :func:`measure_sync` 的同类数字；
        · ``fps`` 用 ``iters * 1000 / wall`` 算，``wall`` 覆盖「整批提交开始 → 全部完成」
          的墙钟时间 ⇒ 这是**聚合吞吐**，与 :func:`measure_sync` 里「中位延迟取倒数」
          得到的 fps 不是同一个量，两者不可直接横向比较。

    返回 ``{"latency_ms", "fps", "wall_ms"}``；若一个回调都没落库，``latency_ms`` 为 nan，
    而 ``wall_ms`` 仍然有效（它只依赖墙钟，不依赖回调）。
    """
    q = ov.AsyncInferQueue(compiled, jobs=nstreams)

    latencies: list[float] = []
    start_times: list[float] = []

    def cb(request, userdata) -> None:
        """AsyncInferQueue 的完成回调：把「提交 → 完成」的驻留时间记进共享列表。

        两点约定：
            · ``userdata`` 必须是**提交时刻**的 ``time.perf_counter()`` ——
              回调只做 ``now - userdata``，漏传或传 None 会在回调线程里抛 TypeError；
            · 回调由 OpenVINO 的工作线程触发，``list.append`` 在 CPython 下是原子的，
              所以这里不加锁；但落库顺序不代表完成顺序，调用方只能取中位数。

        预热期的样本不在这里过滤 —— 由调用方在 ``wait_all()`` 之后 ``latencies.clear()``
        统一丢掉，回调只负责「有一次完成就记一次」。
        """
        latencies.append((time.perf_counter() - userdata) * 1000)

    q.set_callback(cb)

    for _ in range(warmup):
        q.start_async(inputs=[data], userdata=time.perf_counter())
    q.wait_all()
    latencies.clear()

    t_begin = time.perf_counter()
    for i in range(iters):
        start_times.append(time.perf_counter())
        q.start_async(inputs=[data], userdata=start_times[-1])
    q.wait_all()
    wall = (time.perf_counter() - t_begin) * 1000

    med = statistics.median(latencies) if latencies else float("nan")
    return {
        "latency_ms": round(med, 2),
        "fps": round(iters * 1000.0 / wall, 2),
        "wall_ms": round(wall, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    core = ov.Core()
    print(f"OpenVINO {ov.__version__} · devices {core.available_devices}")
    results = {}

    for tag, xml in MODELS.items():
        if not Path(xml).exists():
            print(f"  跳过 {tag}（{xml} 不存在）")
            continue
        data = None
        results[tag] = {}

        # 1) LATENCY hint，同步，等价于 1 stream
        c = core.compile_model(core.read_model(xml), "CPU",
                               {"PERFORMANCE_HINT": "LATENCY"})
        data = build_input(c)
        results[tag]["sync/latency-hint/1-stream"] = measure_sync(c, data, args.iters)

        # 2) LATENCY hint，异步队列深度 1
        results[tag]["async/latency-hint/1-stream"] = measure_async(c, data, args.iters, 1)

        # 3) THROUGHPUT hint，异步，多 stream（AUTO 由 OpenVINO 依据核心数决定）
        ct = core.compile_model(core.read_model(xml), "CPU",
                                {"PERFORMANCE_HINT": "THROUGHPUT"})
        nstreams = ct.get_property("NUM_STREAMS") or 1
        # 注意：输入需要是单张图的 tensor，AsyncInferQueue 会复制填充各 job
        results[tag][f"async/throughput-hint/{nstreams}-streams"] = \
            measure_async(ct, data, args.iters, max(1, int(nstreams)))
        results[tag]["_num_streams"] = int(nstreams)

        for mode, r in results[tag].items():
            if mode.startswith("_"):
                continue
            print(f"  {tag:5s} {mode:38s} {r['latency_ms']:7.2f} ms   {r['fps']:7.2f} FPS")

    Path("results").mkdir(exist_ok=True)
    (Path("results") / "async_bench.json").write_text(json.dumps(results, indent=2))

    lines = ["| Model | Mode | Latency ms | Throughput FPS |", "|---|---|---|---|"]
    for tag, per_model in results.items():
        for mode, r in per_model.items():
            if mode.startswith("_"):
                continue
            lines.append(f"| {tag} | {mode} | {r['latency_ms']:.2f} | {r['fps']:.2f} |")
    print("\n" + "\n".join(lines))
    print("\n已写入 results/async_bench.json")


if __name__ == "__main__":
    main()
