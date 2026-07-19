# Compression results — reference `fp16_baseline`

Video: `Screen Recording 2026-07-10 at 5.43.10 PM.mov`  ·  device: mps  ·  classes: quadcopter

| Run | Quant | Size (GB) | Hit rate | Latency (s) | IoU vs ref | Drift px | Missed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp16_baseline | none | 7.115 | 94% | 4.319 | - | - | - |
| int4_quanto | int4 | 4.224 | 100% | 23.598 | 0.722 | 52.0 | 0 |
| int8_quanto | int8 | 4.138 | 94% | 16.765 | 0.859 | 33.3 | 0 |

*IoU / Drift / Missed are measured against the reference run on frame-matched top boxes. Weight-only quant shrinks footprint but does not speed up MPS inference (no int matmul kernels).*
