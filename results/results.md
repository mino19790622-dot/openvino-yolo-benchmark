# Measured results

- **Host**: Intel(R) Core(TM) i7-8559U CPU @ 2.70 GHz (Coffee Lake, 4C/8T) — MacBook Pro 2018
- **Software**: OpenVINO 2025.4.1 · NNCF 3.2.0 · Python 3.13 · macOS 15.7 (x86_64)
- **Devices detected**: `CPU` only — no OpenVINO GPU plugin on macOS, no NPU on this chip
- **Method**: synchronous inference, 640×640, batch 1, static shapes.
  **3 rounds × 200 iterations** per variant, 10 warm-up runs discarded, median per round,
  then **median-of-medians** (laptop thermal throttling makes single readings unreliable).
- **Accuracy**: COCO val2017 subset, 200 images, 1,444 ground-truth boxes.
  COCO 101-point interpolation, per-class NMS @ IoU 0.7, max 300 detections/image.

## Performance

| Precision | Latency ms | Throughput FPS | Weights MB | Speed-up |
|---|---|---|---|---|
| FP32 | 55.60 | 17.99 | 12.71 | 1.00× |
| FP16 | 53.27 | 18.77 | 6.35 | 1.04× |
| INT8 (PTQ) | 27.99 | 35.73 | 3.29 | **1.99×** |
| INT8 (accuracy-aware) | 31.69 | 31.56 | 3.34 | 1.75× |

Round-to-round spread: FP32 8.36 ms (first round is a cold-start outlier: 63.03 / 55.60 / 54.67),
all others ≤ 1.7 ms.

## Accuracy

| Precision | mAP50-95 | mAP50 | Δ mAP50-95 vs FP32 |
|---|---|---|---|
| FP32 | 0.4035 | 0.5543 | — |
| FP16 | 0.4030 | 0.5541 | −0.0005 |
| INT8 (PTQ) | 0.3455 | 0.5296 | −0.0580 |
| INT8 (accuracy-aware) | 0.3897 | 0.5440 | −0.0138 |

## Sanity check of the evaluation pipeline

Cross-checked against Ultralytics' published YOLOv8n figures on the full COCO val2017:

| | This repo (200-image subset) | Ultralytics (full val2017) |
|---|---|---|
| mAP50-95 | 0.4035 | 0.370 |
| mAP50 | 0.5543 | 0.539 |

mAP50 within 0.015 → the letterbox preprocess, anchor decode, coordinate restoration,
per-class NMS and the COCO `category_id` (1…90, non-contiguous) → YOLO index (0…79) mapping
are all correct.

> NOTE: `coco128` is drawn from COCO **train2017**, which the model was trained on — it reports
> mAP50-95 = 0.4444, noticeably optimistic. Absolute accuracy figures therefore always come from
> val2017; `Coco128Set` is used only for quantisation calibration and smoke tests.

## Reproduce

```bash
python run_benchmark.py --eval-set val2017 --limit 200 --accuracy-control
```

Raw per-round numbers are in `verify.json`.
