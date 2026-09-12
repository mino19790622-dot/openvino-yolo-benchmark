# Measured results

- Host: Intel(R) Core(TM) i7-8559U CPU @ 2.70GHz · Darwin 24.6.0 (x86_64)
- OpenVINO 2025.4.1-20426-82bbf0292c5-releases/2025/4 · devices seen: CPU
- Method: synchronous inference, 200 iterations after 10 warm-up, median

| Precision | Device | Latency ms (median) | Throughput FPS | Weights MB |
|---|---|---|---|---|
| FP32 | CPU | 60.74 | 16.46 | 12.71 |
| FP16 | CPU | 57.07 | 17.52 | 6.35 |
| INT8 | CPU | 33.11 | 30.20 | 3.29 |
| INT8-AWQ | CPU | 34.71 | 28.81 | 3.34 |

| Precision | mAP50-95 | Δ vs FP32 | Weights MB |
|---|---|---|---|
| FP32 | 0.4035 | — | 12.71 |
| FP16 | 0.4030 | -0.0005 | 6.35 |
| INT8 | 0.3455 | -0.0580 | 3.29 |
| INT8-AWQ | 0.3897 | -0.0138 | 3.34 |

Evaluated on **COCO val2017 subset (200 images)**, 1444 ground-truth boxes.
