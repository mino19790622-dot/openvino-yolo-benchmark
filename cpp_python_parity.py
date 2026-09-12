#!/usr/bin/env python3
"""对比同一张图上 C++ 与 Python 推理管线的 host 端开销。

两边做的是完全一样的事：
    letterbox -> BGR2RGB -> /255 -> NCHW -> 推理 -> 解码 -> 逐类 NMS -> 坐标还原
差异只在实现：C++ 侧（benchmark_cpp/yolo_infer_cpp.cpp）全部手写，
Python 侧（yolo_utils.py）用 numpy + OpenCV。

用法：
    .venv/bin/python cpp_python_parity.py --model models/yolov8n_fp32/yolov8n_fp32.xml
"""

from __future__ import annotations

import argparse
import statistics
import struct
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

import yolo_utils as U

ROOT = Path(__file__).parent
EXE = ROOT / "benchmark_cpp" / "build" / "yolo_infer_cpp"


def dump_raw_bgr(im: np.ndarray, path: Path) -> None:
    h, w = im.shape[:2]
    with open(path, "wb") as f:
        f.write(struct.pack("ii", h, w))
        f.write(im.tobytes())


def run_cpp(xml: Path, bin_path: Path, device: str, iters: int) -> dict:
    if not EXE.exists():
        raise SystemExit(f"找不到 {EXE}，请先编译 benchmark_cpp/")
    out = subprocess.run(
        [str(EXE), str(xml), str(bin_path), device, str(iters)],
        capture_output=True, text=True,
    ).stdout
    kv = {}
    for line in out.splitlines():
        if line.startswith("KV "):
            for part in line[3:].split():
                k, _, v = part.partition("=")
                kv[k] = float(v)
    if not kv:
        print(out)
        raise SystemExit("C++ 程序没有输出 KV 行")
    return kv


def run_python(xml: Path, im: np.ndarray, device: str, iters: int) -> dict:
    core = ov.Core()
    compiled = core.compile_model(core.read_model(str(xml)), device)
    oname = compiled.output().get_any_name()
    h, w = im.shape[:2]

    t_pre, t_inf, t_post, t_total, ndet = [], [], [], [], 0
    for it in range(iters):
        t0 = time.perf_counter()
        x, r, pad = U.preprocess(im)
        t1 = time.perf_counter()
        p = compiled([x])[oname]
        t2 = time.perf_counter()
        b, s, l = U.decode(p, r, pad, (w, h))
        t3 = time.perf_counter()
        if it >= 5:
            t_pre.append((t1 - t0) * 1000)
            t_inf.append((t2 - t1) * 1000)
            t_post.append((t3 - t2) * 1000)
            t_total.append((t3 - t0) * 1000)
            ndet = len(b)
    return {
        "pre_ms": statistics.median(t_pre),
        "infer_ms": statistics.median(t_inf),
        "post_ms": statistics.median(t_post),
        "total_ms": statistics.median(t_total),
        "dets": float(ndet),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/yolov8n_fp32/yolov8n_fp32.xml")
    ap.add_argument("--image", default="")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    if args.image:
        im = cv2.imread(args.image)
    else:
        im = U.load_coco128()[0]["image"]
    if im is None:
        raise SystemExit("读不到图片")

    bin_path = Path("/tmp/_yolo_raw_bgr.bin")
    dump_raw_bgr(im, bin_path)

    cpp = run_cpp(Path(args.model), bin_path, args.device, args.iters)
    py = run_python(Path(args.model), im, args.device, args.iters)

    print(f"\n模型 {Path(args.model).name} · 设备 {args.device} · "
          f"{args.iters} 次迭代 (5 次 warm-up 不计) · 中位数\n")
    print(f"{'stage':<14}{'C++ (ms)':>10}{'Python (ms)':>13}{'speed-up':>11}")
    print("-" * 48)
    for k in ("pre_ms", "infer_ms", "post_ms", "total_ms"):
        speed = py[k] / cpp[k] if cpp[k] else float("nan")
        print(f"{k.replace('_ms',''):<14}{cpp[k]:>10.2f}{py[k]:>13.2f}{speed:>10.2f}x")
    print("-" * 48)
    print(f"检测框数量     C++ {int(cpp['dets'])}    Python {int(py['dets'])}"
          f"   {'✅ 一致' if int(cpp['dets']) == int(py['dets']) else '⚠️ 不一致，需检查'}")

    host_cpp = cpp["pre_ms"] + cpp["post_ms"]
    host_py = py["pre_ms"] + py["post_ms"]
    print(f"\nHost 端开销(前+后处理)  C++ {host_cpp:.2f} ms   Python {host_py:.2f} ms"
          f"   → C++ 快 {host_py / host_cpp:.2f}×")
    print(f"其中后处理单项          C++ {cpp['post_ms']:.2f} ms   "
          f"Python {py['post_ms']:.2f} ms   → {py['post_ms'] / cpp['post_ms']:.2f}×")

    out = {
        "cpp": cpp, "python": py,
        "host_speedup": round(host_py / host_cpp, 2),
        "post_speedup": round(py["post_ms"] / cpp["post_ms"], 2),
    }
    Path("results").mkdir(exist_ok=True)
    import json
    (Path("results") / "cpp_python_parity.json").write_text(json.dumps(out, indent=2))
    print(f"\n已写入 results/cpp_python_parity.json")


if __name__ == "__main__":
    main()
