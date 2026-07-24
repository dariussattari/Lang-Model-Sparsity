# Compression results — reference `yolo11n_fp16`

Video: `Screen Recording 2026-07-10 at 5.43.10 PM.mov`  ·  device: mps  ·  classes: person

| Run | Quant | Size (GB) | Hit rate | Latency (s) | IoU vs ref | Drift px | Missed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| yolo11n_fp16 | none | 0.0098 | 94% | 0.0414 | - | - | - |
| yolov8s-worldv2_fp16 | none | 0.6112 | 6% | 0.0281 | - | - | - |
| yolo11n_int4 | int4 | 0.0027 | 0% | 0.1866 | - | - | 32 |
| yolo11n_int8 | int8 | 0.0026 | 0% | 0.0322 | - | - | 32 |

*IoU / Drift / Missed are measured against the reference run on frame-matched top boxes. Weight-only quant shrinks footprint but does not speed up MPS inference (no int matmul kernels).*
