"""Build a task-aware recovery dataset from the deployment video itself.

Structured pruning shifts activation statistics, so the detector must be
fine-tuned to recover (this is the standard Torch-Pruning workflow). Rather than
pull an external labeled drone set, we pseudo-label the quadcopter video with the
unpruned base model (its confident boxes become ground truth) and fine-tune the
pruned model on *that*. This is deliberately aligned with the project thesis:
recover on the narrow deployment distribution, not on a general dataset.

Writes an ultralytics-format dataset under dataset/:
    dataset/images/{train,val}/*.jpg
    dataset/labels/{train,val}/*.txt   # one row: `0 cx cy w h` (normalized)
    dataset/drone.yaml
Frames where the base model is confident -> a labelled positive. A capped share
of no-detection frames are kept as explicit backgrounds (empty label file) to
hold down false positives after fine-tuning.
"""
import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

import bench

VIDEO = "quadcopter.mov"
BASE = "drone_yolov8x.pt"
ROOT = Path("dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=2, help="keep every Nth frame")
    ap.add_argument("--label-conf", type=float, default=0.5,
                    help="min base-model confidence to accept a pseudo-label")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-bg-frac", type=float, default=0.2,
                    help="cap background (no-detection) frames as share of total kept")
    args = ap.parse_args()

    dev = bench.pick_device()
    model = YOLO(BASE)
    model.model.to(dev).eval()

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kept, bg = [], 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            h, w = frame.shape[:2]
            r = model.predict(frame, verbose=False, device=dev, conf=args.label_conf)[0]
            rows = []
            for b in r.boxes:
                x0, y0, x1, y1 = b.xyxy[0].tolist()
                cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
                bw, bh = (x1 - x0) / w, (y1 - y0) / h
                rows.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            is_bg = len(rows) == 0
            kept.append((idx, frame, rows, is_bg))
        idx += 1
    cap.release()

    # cap background frames
    pos = [k for k in kept if not k[3]]
    bgs = [k for k in kept if k[3]]
    max_bg = int(args.max_bg_frac * (len(pos) + 1) / (1 - args.max_bg_frac))
    bgs = bgs[:max_bg]
    data = pos + bgs

    val_stride = max(2, int(round(1 / args.val_frac)))
    n_train = n_val = 0
    for j, (i, frame, rows, is_bg) in enumerate(data):
        split = "val" if j % val_stride == 0 else "train"
        stem = f"f{i:05d}"
        cv2.imwrite(str(ROOT / "images" / split / f"{stem}.jpg"), frame)
        (ROOT / "labels" / split / f"{stem}.txt").write_text("\n".join(rows))
        if split == "train":
            n_train += 1
        else:
            n_val += 1

    yaml = (ROOT / "drone.yaml")
    yaml.write_text(
        f"path: {ROOT.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: drone\n")
    print(f"positives={len(pos)} backgrounds={len(bgs)} "
          f"-> train={n_train} val={n_val}\nwrote {yaml}")


if __name__ == "__main__":
    main()
