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
    shape = list(compiled.input().get_shape())
    rng = np.random.default_rng(0)
    return rng.random(shape, dtype=np.float32)


def measure_sync(compiled, data, iters: int, warmup: int = 10) -> dict:
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
    """用一个深度为 nstreams 的 AsyncInferQueue 提交 iters 次，测总墙钟吞吐与单请求延迟。"""
    q = ov.AsyncInferQueue(compiled, jobs=nstreams)

    latencies: list[float] = []
    start_times: list[float] = []

    def cb(request, userdata) -> None:
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
    for tag in results:
        for mode, r in results[tag].items():
            if mode.startswith("_"):
                continue
            lines.append(f"| {tag} | {mode} | {r['latency_ms']:.2f} | {r['fps']:.2f} |")
    print("\n" + "\n".join(lines))
    print("\n已写入 results/async_bench.json")


if __name__ == "__main__":
    main()
