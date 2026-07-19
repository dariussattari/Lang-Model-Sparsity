# Compression results — reference `fp16_baseline`

Video: `Screen Recording 2026-07-10 at 5.43.10 PM.mov`  ·  device: mps  ·  classes: quadcopter

| Run | Quant | Size (GB) | Hit rate | Latency (s) | IoU vs ref | Drift px | Missed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp16_baseline | none | 5.65 | 94% | 1.132 | - | - | - |
| int4_quanto | int4 | 3.446 | 91% | 13.422 | 0.833 | 26.5 | 1 |
| int8_quanto | int8 | 3.38 | 94% | 3.031 | 0.973 | 5.3 | 0 |

*IoU / Drift / Missed are measured against the reference run on frame-matched top boxes. Weight-only quant shrinks footprint but does not speed up MPS inference (no int matmul kernels).*
