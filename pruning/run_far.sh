#!/usr/bin/env bash
# Warm-start the recovered pruned models on the near+far dataset (imgsz 768 so
# small/distant drones are resolvable). Continues from the close-only checkpoints.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

echo "=== [1/2] prune 50% -> near+far (warm-start) ==="
python finetune.py --ratio 0.5 \
  --init-from runs/drone_yolov8x_prune50_ft/drone_yolov8x_prune50_ft.pt \
  --data dataset_far/drone.yaml --epochs 20 --imgsz 768 --batch 8 --lr0 1e-3 \
  --tag drone_yolov8x_prune50_ft_far

echo "=== [2/2] prune 75% -> near+far (warm-start) ==="
python finetune.py --ratio 0.75 \
  --init-from runs/drone_yolov8x_prune75_ft/drone_yolov8x_prune75_ft.pt \
  --data dataset_far/drone.yaml --epochs 20 --imgsz 768 --batch 8 --lr0 1e-3 \
  --tag drone_yolov8x_prune75_ft_far

echo "=== FAR TRAINING DONE ==="
