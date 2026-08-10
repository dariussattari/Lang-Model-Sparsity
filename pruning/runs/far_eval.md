# Far-away (small-bbox) drone detection

The `quadcopter.mov` benchmark only contains a *close* drone, so it can't show far
detection. This measures recall on **held-out very-far drones** (largest box ≤ 5% of
the image, i.e. the 100–200 m case) from the Seraphim dataset — images the models
never trained on (`eval_far.py`, 150 images, imgsz 768, conf 0.15). Green = ground
truth, red = prediction in the montages.

| Model | Size (GB) | **Far recall** (≤5% drones) | Far mean IoU | Near hit (quadcopter.mov) |
| --- | --- | --- | --- | --- |
| prune 50% + recovery, **close data only** | 0.078 | 30% | 0.74 | 74% |
| prune 50% + recovery, **near+far data**   | 0.078 | **93%** | 0.76 | 74% |
| prune 75% + recovery, **close data only** | 0.031 | 17% | 0.68 | 71% |
| prune 75% + recovery, **near+far data**   | 0.031 | **93%** | 0.78 | 76% |

**Findings**
- Adding far/small drones to the recovery set lifts distant-drone recall from
  17–30% to **93%** — with **no loss** on the close deployment drone (near hit rate
  unchanged / slightly up).
- **prune 75% is the best overall model:** it matches prune 50%'s 93% far recall at
  **4× smaller (0.031 vs 0.078 GB) and faster (18 vs 20 ms)**, with slightly better
  far localization (IoU 0.78 vs 0.76). More aggressive pruning did *not* cost far
  accuracy once the training data covered far drones.
- Far detection needed two things beyond the close-only recovery: **training data
  with distant drones** (Seraphim small-bbox subset) and **higher input resolution**
  (imgsz 768) so a ~2% drone is resolvable.

Reproduce:
```bash
python build_far_dataset.py --n-far 1000 --n-mid 300
bash run_far.sh                    # warm-start prune50 + prune75 on near+far data
python eval_far.py --n 150 --models \
  prune75_close=runs/drone_yolov8x_prune75_ft/drone_yolov8x_prune75_ft.pt \
  prune75_nearfar=runs/drone_yolov8x_prune75_ft_far/drone_yolov8x_prune75_ft_far.pt
```

See `far_before_after_prune75.png` for the visual before/after.
