# Cross-model compression report

Video: `Screen Recording 2026-07-10 at 5.43.10 PM.mov`

## Per-model (deltas vs each model's own fp16)

| Model | Quant | Size (GB) | Hit rate | Latency (s) | Δ size | Δ latency |
| --- | --- | --- | --- | --- | --- | --- |
| paligemma | none | 5.65 | 94% | 1.132 | - | - |
| paligemma | int8 | 3.38 | 94% | 3.031 | -40% | +168% |
| paligemma | int4 | 3.446 | 91% | 13.422 | -39% | +1086% |
| locateanything | none | 7.115 | 94% | 4.319 | - | - |
| locateanything | int8 | 4.138 | 94% | 16.765 | -42% | +288% |
| locateanything | int4 | 4.224 | 100% | 23.598 | -41% | +446% |

## Cross-model (same quant level, side by side)


### quant = none

| Model | Size (GB) | Hit rate | Latency (s) | Δ size vs paligemma | Δ latency vs paligemma |
| --- | --- | --- | --- | --- | --- |
| paligemma | 5.65 | 94% | 1.132 | - | - |
| locateanything | 7.115 | 94% | 4.319 | +26% | +282% |

### quant = int8

| Model | Size (GB) | Hit rate | Latency (s) | Δ size vs paligemma | Δ latency vs paligemma |
| --- | --- | --- | --- | --- | --- |
| paligemma | 3.38 | 94% | 3.031 | - | - |
| locateanything | 4.138 | 94% | 16.765 | +22% | +453% |

### quant = int4

| Model | Size (GB) | Hit rate | Latency (s) | Δ size vs paligemma | Δ latency vs paligemma |
| --- | --- | --- | --- | --- | --- |
| paligemma | 3.446 | 91% | 13.422 | - | - |
| locateanything | 4.224 | 100% | 23.598 | +23% | +76% |

## Cross-model detection agreement (paligemma vs locateanything, fp16)

- frames both detected: 32
- mean IoU of top boxes: 0.655
- mean centroid drift: 64.2 px

*Low IoU here doesn't mean either model is wrong — the models localize differently; it quantifies how interchangeable they are.*
