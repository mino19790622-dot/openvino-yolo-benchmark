#!/usr/bin/env python3
"""
OpenVINO YOLOv8 Benchmark — 完全不依赖 PyTorch 的版本
=====================================================

为什么不让用 ultralytics/YOLO PyTorch 导出？
    本机是 macOS x86_64（Intel Core i7-8559U），PyTorch 官方不为该平台提供 wheel，
    且 OpenVINO 在此系统上只识别到 CPU（无 GPU 插件 / 无 NPU）。
    所以整条链路改为：官方 yolov8n.onnx → ovc 转 IR → NNCF INT8 PTQ → OpenVINO Runtime。

产出（全部为本机实测）：
    results/results.md   —— Markdown 结果表
    results/results.json —— 原始数据

用法：
    # 全量：量化 + 精度评估 + 性能基准
    .venv/bin/python run_benchmark.py

    # 只跑性能（跳过 mAP，快）
    .venv/bin/python run_benchmark.py --skip-accuracy
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import openvino as ov

import yolo_utils as U

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

DEVICES = ["CPU", "GPU", "NPU"]  # 本机只会命中 CPU，脚本会按实际情况裁剪


def hr(t: str) -> None:
    print("\n" + "=" * 70)
    print("  " + t)
    print("=" * 70)


def cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:                # noqa: BLE001
            pass
    elif platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
        except Exception:                # noqa: BLE001
            pass
    return platform.processor() or "unknown"


# --------------------------------------------------------------------------
# 1. ONNX -> IR
# --------------------------------------------------------------------------
def build_variants(size: int = 640, accuracy_control: bool = False):
    """生成 FP32 / FP16 / INT8 / INT8-AWQ 四个 IR。返回 {tag: xml_path}"""
    import nncf

    MODELS.mkdir(exist_ok=True)
    onnx = ASSETS / "yolov8n.onnx"
    if not onnx.exists():
        raise SystemExit(f"缺少 {onnx}，请先下载")

    core = ov.Core()
    out: dict[str, Path] = {}

    hr("步骤 1/3  构建 FP32 / FP16 IR")
    fp32 = MODELS / "yolov8n_fp32" / "yolov8n_fp32.xml"
    if not fp32.exists():
        subprocess.check_call([
            "ovc", str(onnx),
            "--output_model", str(fp32),
            "--compress_to_fp16=False",
        ])
    out["FP32"] = fp32

    fp16_xml = MODELS / "yolov8n_fp16" / "yolov8n_fp16.xml"
    if not fp16_xml.exists():
        # ovc 默认 compress_to_fp16=True，正好产出 FP16 IR
        subprocess.check_call([
            "ovc", str(onnx),
            "--output_model", str(fp16_xml),
        ])
    out["FP16"] = fp16_xml

    hr("步骤 2/3  NNCF INT8 post-training quantisation")
    int8 = MODELS / "yolov8n_int8" / "yolov8n_int8.xml"
    if not int8.exists():
        samples = U.load_coco128()
        calib_items = [s["image"] for s in samples]

        def transform(im: np.ndarray):
            x, _, _ = U.preprocess(im, size)
            return x

        calib = nncf.Dataset(calib_items, transform)
        q = nncf.quantize(
            ov.Core().read_model(fp32),
            calib,
            subset_size=len(calib_items),
            preset=nncf.QuantizationPreset.PERFORMANCE,
            target_device=nncf.TargetDevice.CPU,
        )
        ov.save_model(q, str(int8))
    out["INT8"] = int8

    # ---- accuracy-aware quantisation（rank-based 敏感度回退）----
    awq = MODELS / "yolov8n_int8_awq" / "yolov8n_int8_awq.xml"
    if accuracy_control and not awq.exists():
        hr("NNCF accuracy-aware quantisation (把敏感层回退到 FP16/FP32)")
        samples = U.load_coco128()
        # 三份数据严格分开：校准 / 敏感度调参 / 最终测试(外部 val2017)
        calib_items = [s["image"] for s in samples[32:]]
        tune_items = samples[:32]

        def transform(im: np.ndarray):
            x, _, _ = U.preprocess(im, size)
            return x

        calib = nncf.Dataset(calib_items, transform)
        tune_ds = nncf.Dataset(tune_items, lambda s: U.preprocess(s["image"], size)[0])

        def validation_fn(model, dataset) -> float:
            # NNCF 传入的 model_for_inference 可能是 ov.Model 也可能已经是
            # ov.CompiledModel，后者再 compile 一次会走错重载被当成路径。
            if isinstance(model, ov.CompiledModel):
                compiled = model
            else:
                compiled = ov.Core().compile_model(model, "CPU")
            oname = compiled.output().get_any_name()
            preds = []
            for s in tune_items:
                im = s["image"]
                x, r, pad = U.preprocess(im, size)
                p = compiled([x])[oname]
                b, sc, l = U.decode(p, r, pad, (im.shape[1], im.shape[0]))
                preds.append(dict(gt_boxes=s["gt_boxes"], gt_classes=s["gt_classes"],
                                  boxes=b, scores=sc, labels=l))
            return U.compute_coco_map(preds)["mAP50-95"]

        q = nncf.quantize_with_accuracy_control(
            ov.Core().read_model(fp32),
            calib,
            validation_dataset=tune_ds,
            validation_fn=validation_fn,
            max_drop=0.01,
            drop_type=nncf.DropType.ABSOLUTE,
            preset=nncf.QuantizationPreset.PERFORMANCE,
            target_device=nncf.TargetDevice.CPU,
        )
        ov.save_model(q, str(awq))
    if accuracy_control:
        out["INT8-AWQ"] = awq

    for tag, p in out.items():
        mb = (p.with_suffix(".bin").stat().st_size) / 1e6
        print(f"  {tag:9s} -> {p.name:28s} weights {mb:.2f} MB")
    return out


# --------------------------------------------------------------------------
# 2. 性能
# --------------------------------------------------------------------------
def _detect_devices(core: ov.Core) -> list[str]:
    found = [d.replace(" (", "|") for d in core.available_devices]
    devs = [d for d in DEVICES if any(d in f for f in found)]
    return devs


def measure(xml: Path, device: str, iters: int, warmup: int = 10) -> dict | None:
    """同步推理计时（等价于 benchmark_app -api sync -nstreams 1）。"""
    core = ov.Core()
    try:
        compiled = core.compile_model(core.read_model(str(xml)), device)
    except Exception as exc:                       # noqa: BLE001
        print(f"  !! {device} 编译失败: {str(exc).splitlines()[0][:150]}")
        return None

    inp = compiled.input()
    shape = list(inp.get_shape())
    rng = np.random.default_rng(0)
    data = rng.random(shape, dtype=np.float32)
    req = compiled.create_infer_request()

    for _ in range(warmup):
        req.set_input_tensor(ov.Tensor(data))
        req.infer()

    samples = []
    for _ in range(iters):
        req.set_input_tensor(ov.Tensor(data))
        t0 = time.perf_counter()
        req.infer()
        samples.append((time.perf_counter() - t0) * 1000.0)

    a = np.asarray(samples)
    return {
        "device": device,
        "latency_ms": round(float(np.median(a)), 2),
        "latency_mean_ms": round(float(a.mean()), 2),
        "latency_p95_ms": round(float(np.percentile(a, 95)), 2),
        "fps": round(1000.0 / float(np.median(a)), 2),
    }


def evaluate(xml: Path, device: str, evaluator):
    """跑完整评测集，返回 mAP。"""
    core = ov.Core()
    compiled = core.compile_model(core.read_model(str(xml)), device)
    oname = compiled.output().get_any_name()
    preds = []
    for im, meta in evaluator.images():
        x, r, pad = U.preprocess(im)
        p = compiled([x])[oname]
        b, s, l = U.decode(p, r, pad, (im.shape[1], im.shape[0]))
        preds.append(dict(gt_boxes=meta["gt_boxes"], gt_classes=meta["gt_classes"],
                          boxes=b, scores=s, labels=l))
    return U.compute_coco_map(preds), preds


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200, help="每组bench迭代次数")
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--eval-set", default="coco128", choices=["coco128", "val2017"],
                    help="coco128=128张(来自train2017)；val2017=COCO官方验证子集")
    ap.add_argument("--limit", type=int, default=300, help="val2017 抽样图片数")
    ap.add_argument("--accuracy-control", action="store_true",
                    help="额外跑 NNCF accuracy-aware quantisation（较慢，但能挽回精度）")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    hr("环境")
    core = ov.Core()
    devices = _detect_devices(core)
    env = {
        "openvino": ov.__version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": cpu_name(),
        "devices_seen": core.available_devices,
        "devices_used": devices,
    }
    for k, v in env.items():
        print(f"  {k:14s}: {v}")

    variants = build_variants(accuracy_control=args.accuracy_control)

    # ---- 评测 ----
    acc = {}
    if not args.skip_accuracy:
        hr(f"步骤 3/3  精度评估 (mAP on {args.eval_set})")
        from coco_eval_set import Coco128Set, make_eval_set
        try:
            evaluator = make_eval_set(args.eval_set, limit=args.limit)
        except Exception as exc:                    # noqa: BLE001
            print(f"  评测集 unavailable({exc})，回退到 coco128")
            evaluator = Coco128Set()
        print(f"  评测集: {evaluator.name()}")

        print(f"  图片数 {evaluator.n()}   GT 框 {evaluator.n_gt()}")
        for tag, xml in variants.items():
            res, _ = evaluate(xml, "CPU", evaluator)
            acc[tag] = round(res["mAP50-95"], 4)
            acc[tag + "_50"] = round(res["mAP50"], 4)
            print(f"  {tag:9s} mAP50-95={res['mAP50-95']:.4f}  mAP50={res['mAP50']:.4f}")
        base = acc["FP32"]
        print()
        for tag in variants:
            if tag != "FP32":
                acc[f"delta_{tag}"] = round(acc[tag] - base, 4)
                print(f"  Δ vs FP32   {tag:9s} = {acc[f'delta_{tag}']:+.4f}")

    # ---- 性能 ----
    hr("性能基准")
    runs = []
    for tag, xml in variants.items():
        for dev in devices:
            r = measure(xml, dev, args.iters)
            if r:
                r["precision"] = tag
                r["model_mb"] = round(xml.with_suffix(".bin").stat().st_size / 1e6, 2)
                runs.append(r)
                print(f"  {tag:9s} {dev:4s} {r['latency_ms']:7.2f} ms  {r['fps']:7.2f} FPS  "
                      f"{r['model_mb']:6.2f} MB")

    payload = {"env": env, "accuracy": acc, "runs": runs}
    (RESULTS / "results.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Measured results",
        "",
        f"- Host: {env['cpu']} · {env['platform']}",
        f"- OpenVINO {env['openvino']} · devices seen: {', '.join(env['devices_seen'])}",
        f"- Method: synchronous inference, {args.iters} iterations after {10} warm-up, median",
        "",
        "| Precision | Device | Latency ms (median) | Throughput FPS | Weights MB |",
        "|---|---|---|---|---|",
    ]
    for r in runs:
        lines.append(f"| {r['precision']} | {r['device']} | {r['latency_ms']:.2f} | "
                     f"{r['fps']:.2f} | {r['model_mb']:.2f} |")
    if acc:
        lines += ["", f"| Precision | mAP50-95 | Δ vs FP32 | Weights MB |", "|---|---|---|---|"]
        sizes = {r["precision"]: r["model_mb"] for r in runs}
        for tag in variants:
            d = "—" if tag == "FP32" else f"{acc[f'delta_{tag}']:+.4f}"
            lines.append(f"| {tag} | {acc[tag]:.4f} | {d} | {sizes.get(tag, '-')} |")
        lines.append("")
        lines.append(f"Evaluated on **{evaluator.name()}**, {evaluator.n_gt()} ground-truth boxes.")
    (RESULTS / "results.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\n已写入 {RESULTS/'results.md'}")


if __name__ == "__main__":
    main()
