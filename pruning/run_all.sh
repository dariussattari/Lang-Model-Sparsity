#!/usr/bin/env bash
# Drive every remaining run sequentially. Baseline + naive-prune runs already exist.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

echo "=== [1/4] quant-only baseline on the drone model (does quant collapse it too?) ==="
python run_experiment.py --ratios 0 --quant-after int8

echo "=== [2/4] prune 50% + recovery fine-tune ==="
python finetune.py --ratio 0.5 --epochs 40

echo "=== [3/4] prune 75% + recovery fine-tune ==="
python finetune.py --ratio 0.75 --epochs 60

echo "=== [4/4] prune 50% (recovered) + int8 quant combo ==="
python finetune.py --ratio 0.5 --from-weights runs/drone_yolov8x_prune50_ft/drone_yolov8x_prune50_ft.pt --quant-after int8

echo "=== ALL DONE ==="
