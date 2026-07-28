"""Assemble a near+far drone dataset for retraining.

Problem: the first recovery set came only from the close-up quadcopter video, so the
pruned models never saw a distant drone. This mixes in *small-bbox* (= far away)
drones from the Seraphim drone dataset (test split, single 'drone' class) with the
existing close-up deployment frames, so the retrained model handles both.

Steps:
  1. extract seraphim/images.zip + labels.zip -> seraphim/test/{images,labels}
  2. bucket every Seraphim image by its largest drone box (fraction of image size):
     far <= --far-thr, mid <= --mid-thr, else near
  3. build dataset_far/ = all close-up frames from dataset/ (oversampled)
     + capped Seraphim far (+ some mid) drones, split train/val
     with far examples guaranteed in val so val mAP reflects far detection.
"""
import argparse
import random
import shutil
import zipfile
from pathlib import Path

SERA = Path("seraphim")
CLOSE = Path("dataset")          # existing close-up pseudo-labels from quadcopter.mov
OUT = Path("dataset_far")
random.seed(0)


def unzip(zp, dest):
    if dest.exists() and any(dest.iterdir()):
        return
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(dest)


def find_pairs(img_root, lbl_root):
    """Map stem -> (image_path, label_path) for images that have a label."""
    imgs = {p.stem: p for p in img_root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    pairs = {}
    for lp in lbl_root.rglob("*.txt"):
        if lp.stem in imgs:
            pairs[lp.stem] = (imgs[lp.stem], lp)
    return pairs


def max_box_frac(label_path):
    """Largest max(w,h) over boxes in a YOLO label file (0 if empty)."""
    best = 0.0
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        w, h = float(parts[3]), float(parts[4])
        best = max(best, max(w, h))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--far-thr", type=float, default=0.10,
                    help="largest box <= this fraction of image => 'far'")
    ap.add_argument("--mid-thr", type=float, default=0.25)
    ap.add_argument("--n-far", type=int, default=900, help="cap on far images")
    ap.add_argument("--n-mid", type=int, default=250, help="cap on mid images")
    ap.add_argument("--close-repeat", type=int, default=3,
                    help="duplicate each close-up frame N times to balance vs Seraphim")
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    unzip(SERA / "images.zip", SERA / "test/images")
    unzip(SERA / "labels.zip", SERA / "test/labels")
    pairs = find_pairs(SERA / "test/images", SERA / "test/labels")
    print(f"seraphim labeled images: {len(pairs)}")

    far, mid = [], []
    for stem, (ip, lp) in pairs.items():
        f = max_box_frac(lp)
        if f <= 0:
            continue
        if f <= args.far_thr:
            far.append((ip, lp, f))
        elif f <= args.mid_thr:
            mid.append((ip, lp, f))
    far.sort(key=lambda x: x[2])          # smallest (farthest) first
    mid.sort(key=lambda x: x[2])
    print(f"far(<= {args.far_thr})={len(far)}  mid(<= {args.mid_thr})={len(mid)}")
    far = far[:args.n_far]
    mid = mid[:args.n_mid]

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = OUT / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    def place(src_img, src_lbl, split, tag):
        stem = f"{tag}_{src_img.stem}"
        shutil.copy(src_img, OUT / "images" / split / f"{stem}{src_img.suffix}")
        shutil.copy(src_lbl, OUT / "labels" / split / f"{stem}.txt")

    n = {"train": 0, "val": 0}
    # Seraphim far/mid, far guaranteed into val
    for group, tag in ((far, "sera_far"), (mid, "sera_mid")):
        for i, (ip, lp, _) in enumerate(group):
            split = "val" if i % int(round(1 / args.val_frac)) == 0 else "train"
            place(ip, lp, split, tag)
            n[split] += 1

    # close-up deployment frames from dataset/ (both splits), oversampled into train
    close_pairs = find_pairs(CLOSE / "images", CLOSE / "labels")
    for j, (stem, (ip, lp)) in enumerate(close_pairs.items()):
        base_split = "val" if j % int(round(1 / args.val_frac)) == 0 else "train"
        reps = args.close_repeat if base_split == "train" else 1
        for r in range(reps):
            place(ip, lp, base_split, f"close{r}")
            n[base_split] += 1

    (OUT / "drone.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: drone\n")
    print(f"dataset_far: train={n['train']} val={n['val']}  -> {OUT/'drone.yaml'}")


if __name__ == "__main__":
    main()
