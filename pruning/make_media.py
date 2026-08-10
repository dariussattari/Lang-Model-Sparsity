"""Build before/after comparison images from the annotated run videos.

Pulls the same frame index out of each run's *_annotated.mp4, adds a caption bar
(run label + hit/size/latency from run_stats.json), and tiles them into a single
PNG so the pruning effect is visible at a glance. Pure OpenCV — no model needed.
"""
import argparse
import json
from pathlib import Path

import cv2

RUNS = Path("runs")
VIDEO_STEM = "quadcopter"


def caption_for(run):
    sp = RUNS / run / "run_stats.json"
    if not sp.exists():
        return run
    s = json.loads(sp.read_text())
    return (f"{run}   hit {s['hit']}/{s['n_passes']} ({s['hit_rate']*100:.0f}%)   "
            f"{s['footprint_gb']:.3f} GB   {s['avg_latency_s']*1000:.0f} ms")


def grab(run, frame_idx):
    mp4 = RUNS / run / f"{VIDEO_STEM}_annotated.mp4"
    if not mp4.exists():
        return None
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def with_caption(img, text, w):
    import cv2
    h = img.shape[0]
    scale = w / img.shape[1]
    img = cv2.resize(img, (w, int(h * scale)))
    bar = 34
    canvas = cv2.copyMakeBorder(img, bar, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(canvas, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1,
                cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=380, help="frame index to sample")
    ap.add_argument("--runs", nargs="+",
                    default=["drone_yolov8x_fp16", "drone_yolov8x_prune50",
                             "drone_yolov8x_prune50_ft"],
                    help="runs to tile (left->right)")
    ap.add_argument("--tile-w", type=int, default=520)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tiles = []
    for run in args.runs:
        f = grab(run, args.frame)
        if f is None:
            print(f"  skip {run} (no annotated video)")
            continue
        tiles.append(with_caption(f, caption_for(run), args.tile_w))
    if not tiles:
        print("no tiles"); return
    h = min(t.shape[0] for t in tiles)
    tiles = [t[:h] for t in tiles]
    strip = cv2.hconcat(tiles)
    out = args.out or str(RUNS / f"before_after_f{args.frame}.png")
    cv2.imwrite(out, strip)
    print(f"wrote {out}  ({strip.shape[1]}x{strip.shape[0]})")


if __name__ == "__main__":
    main()
