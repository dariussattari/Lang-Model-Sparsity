"""Evaluate far-away (small-bbox) drone recall on held-out Seraphim images.

The quadcopter.mov benchmark only shows the *close* deployment drone, so it can't
reveal far-detection gains. This measures recall on genuinely distant drones
(largest box <= --thr of the image) that were NOT used in training, and compares
models (e.g. close-only vs near+far retrained). Also writes a montage PNG.

    python eval_far.py --models \
        close_only=runs/drone_yolov8x_prune75_ft/drone_yolov8x_prune75_ft.pt \
        near_far=runs/drone_yolov8x_prune75_ft_far/drone_yolov8x_prune75_ft_far.pt
"""
import argparse
import random
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

import bench
import prune_lib  # noqa: F401  keep C2f_v2 importable for unpickling pruned checkpoints

SERA_IMG = Path("seraphim/test/images")
SERA_LBL = Path("seraphim/test/labels")
FAR_OUT = Path("dataset_far")
random.seed(1)


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def gt_boxes(lbl, w, h):
    out = []
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        out.append([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h, max(bw, bh)])
    return out


def held_out_far(thr, n):
    """Very-far images not used in dataset_far training."""
    used = set()
    for p in (FAR_OUT / "images/train").glob("*"):
        used.add(p.stem.replace("sera_far_", "").replace("sera_mid_", ""))
    imgs = {p.stem: p for p in SERA_IMG.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    picks = []
    for lp in SERA_LBL.rglob("*.txt"):
        st = lp.stem
        if st in used or st not in imgs:
            continue
        mx = 0.0
        for line in lp.read_text().splitlines():
            q = line.split()
            if len(q) >= 5:
                mx = max(mx, max(float(q[3]), float(q[4])))
        if 0 < mx <= thr:
            picks.append((imgs[st], lp))
    random.shuffle(picks)
    return picks[:n]


def load_model(spec, dev):
    if spec.endswith(".pt") and "prune" in spec and Path(spec).exists():
        ckpt = torch.load(spec, map_location=dev, weights_only=False)
        mod = ckpt.get("model") or ckpt.get("ema") if isinstance(ckpt, dict) else None
        if mod is not None and hasattr(mod, "modules"):
            y = YOLO("drone_yolov8x.pt")
            y.model = mod.to(dev).float().eval()
            y.model.names = ckpt.get("names", {0: "drone"})
            return y
    return YOLO(spec)


def eval_model(y, samples, dev, imgsz, conf):
    hit = 0
    ious = []
    for ip, lp in samples:
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        gts = gt_boxes(lp, w, h)
        r = y.predict(img, verbose=False, device=dev, conf=conf, imgsz=imgsz)[0]
        preds = [b.xyxy[0].tolist() for b in r.boxes]
        matched = False
        for g in gts:
            best = max((iou(g[:4], p) for p in preds), default=0.0)
            if best >= 0.2:
                matched = True
                ious.append(best)
        hit += 1 if matched else 0
    return {"n": len(samples), "recall": round(hit / max(len(samples), 1), 3),
            "mean_iou": round(sum(ious) / len(ious), 3) if ious else 0.0}


def montage(y, samples, dev, imgsz, conf, out, cols=4, rows=3):
    tiles = []
    for ip, lp in samples[:cols * rows]:
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        for g in gt_boxes(lp, w, h):           # GT green
            cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 220, 0), 2)
        r = y.predict(img, verbose=False, device=dev, conf=conf, imgsz=imgsz)[0]
        for b in r.boxes:                      # pred red
            x0, y0, x1, y1 = map(int, b.xyxy[0].tolist())
            cv2.rectangle(img, (x0, y0), (x1, y1), (72, 68, 230), 2)
        tiles.append(cv2.resize(img, (360, 360)))
    if not tiles:
        return
    grid = []
    for r0 in range(0, len(tiles), cols):
        row = tiles[r0:r0 + cols]
        while len(row) < cols:
            row.append(row[-1] * 0)
        grid.append(cv2.hconcat(row))
    cv2.imwrite(out, cv2.vconcat(grid))
    print(f"  montage -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path.pt ...")
    ap.add_argument("--thr", type=float, default=0.05, help="very-far: largest box <= thr")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--conf", type=float, default=0.15)
    args = ap.parse_args()

    dev = bench.pick_device()
    samples = held_out_far(args.thr, args.n)
    print(f"held-out very-far images (<= {args.thr}): {len(samples)}  imgsz={args.imgsz} conf={args.conf}")
    for spec in args.models:
        name, path = spec.split("=", 1)
        y = load_model(path, dev)
        res = eval_model(y, samples, dev, args.imgsz, args.conf)
        print(f"{name:14s} recall={res['recall']*100:.0f}%  mean_iou={res['mean_iou']}  (n={res['n']})")
        montage(y, samples, dev, args.imgsz, args.conf, str(Path("runs") / f"far_montage_{name}.png"))


if __name__ == "__main__":
    main()
