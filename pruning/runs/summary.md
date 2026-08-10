# Pruning experiment — drone YOLOv8x on the quadcopter video

Base model: `doguilmak/Drone-Detection-YOLOv8x` (single class `drone`). Same video, sampling, and schema as the `yolo/` and VLM baselines. MPS, conf 0.25.

| Run | Size (GB) | vs base | Params ↓ | Hit rate | IoU vs base | Drift px | Latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base (fp16, unpruned) | 0.2541 | 100% | - | 28/34 (82%) | — | — | 48 |
| quant int8 (no prune) | 0.0640 | 25% | - | 34/34 (100%) | 0.26 | 223 | 71 |
| prune 25%, no recovery | 0.1521 | 60% | -40% | 0/34 (0%) | — | — | 31 |
| prune 50%, no recovery | 0.0776 | 31% | -70% | 0/34 (0%) | — | — | 19 |
| prune 75%, no recovery | 0.0308 | 12% | -88% | 0/34 (0%) | — | — | 13 |
| prune 50% + recovery (close) | 0.0776 | 31% | -70% | 25/34 (74%) | 0.93 | 10 | 20 |
| prune 75% + recovery (close) | 0.0308 | 12% | -88% | 24/34 (71%) | 0.93 | 10 | 14 |
| prune 50% + recovery (near+far) | 0.0776 | 31% | - | 25/34 (74%) | 0.93 | 10 | 20 |
| prune 75% + recovery (near+far) | 0.0308 | 12% | -88% | 26/34 (76%) | 0.93 | 10 | 18 |
| prune 50% + recovery + int8 | 0.0196 | 8% | -70% | 34/34 (100%) | 0.28 | 200 | 33 |

**Read the IoU column, not just hit rate.** Hit rate counts a pass as a hit if the model emits *any* box — so it is fooled by degenerate outputs. Naive **int8** scores a perfect hit rate yet ~0.26 IoU and >200 px centroid drift: it spams a fixed corner box on every frame and never tracks the drone (the project's own thesis — a metric looks fine while the capability is silently destroyed). No-recovery **pruning** collapses the other way, to 0 detections, because channel removal shifts the head's feature statistics. A short **recovery fine-tune** on the pseudo-labeled deployment video restores real, well-localized boxes (high IoU) at a fraction of the base size and latency.
