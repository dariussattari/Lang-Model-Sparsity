"""YOLO (CNN) open/closed-vocab detection runner.

Third model family in the model-agnostic benchmark. Writes the SAME results
schema as paligemma/ and locateanything/ (see ../analysis/RESULTS_SCHEMA.md) so
the shared ../analysis/ scripts compare all of them without changes.

Two model kinds, auto-detected from the weights name:
  - *-world*  -> YOLO-World, OPEN-vocab: set_classes(<your classes>), detects them
  - anything else -> stock COCO detector (yolov8n, yolo11n, ...): CLOSED 80-class
    vocab. "quadcopter"/"drone" are NOT COCO classes, so these score 0 on the drone
    unless you fine-tune. That gap is exactly what motivates the CNN research track.

Usage:
    python detect.py --video clip.mov quadcopter                       # yolov8s-worldv2
    python detect.py --video clip.mov quadcopter --model yolo11n.pt    # stock COCO
    python detect.py --video clip.mov quadcopter --quant int8
    python detect.py photo.jpg person                                  # single image
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

DEFAULT_MODEL = "yolov8s-worldv2.pt"   # open-vocab; detects arbitrary text classes


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_footprint_gb(module):
    seen = {}
    for t in list(module.parameters()) + list(module.buffers()):
        sub = [getattr(t, a) for a in ("_data", "_scale", "_shift")
               if isinstance(getattr(t, a, None), torch.Tensor)]
        for s in (sub or [t]):
            try:
                st = s.untyped_storage()
                seen[st.data_ptr()] = st.nbytes()
            except Exception:
                seen[id(s)] = s.nelement() * s.element_size()
    return round(sum(seen.values()) / 1024**3, 4)


def load_model(weights, classes, quant, device):
    from ultralytics import YOLO
    print(f"Loading {weights} on {device} ...", flush=True)
    yolo = YOLO(weights)
    is_world = "world" in Path(weights).stem.lower()
    if is_world:
        yolo.set_classes(classes)
    else:
        vocab = set(yolo.names.values())
        missing = [c for c in classes if c not in vocab]
        if missing:
            print(f"  NOTE: {missing} not in this model's {len(vocab)}-class COCO "
                  f"vocab; it cannot detect them without fine-tuning.", flush=True)

    n_quantized = 0
    if quant != "none":
        from optimum.quanto import quantize, freeze, qint8, qint4
        qtype = {"int8": qint8, "int4": qint4}[quant]
        print(f"Quantizing weights to {quant} via quanto ...", flush=True)
        quantize(yolo.model, weights=qtype)
        freeze(yolo.model)
        # count how many layers actually became quanto tensors (Conv2d support varies)
        n_quantized = sum(1 for m in yolo.model.modules()
                          if type(m).__name__.startswith(("QConv", "QLinear")))

    yolo.model.to(device)
    fp = model_footprint_gb(yolo.model)
    print(f"Model footprint: {fp} GB ({quant}, {n_quantized} quantized layers)", flush=True)
    return yolo, is_world, fp, n_quantized


def detect(yolo, is_world, image_bgr, classes, device, conf):
    """Return list of {label, box[x0,y0,x1,y1] pixels} for requested classes."""
    res = yolo.predict(image_bgr, verbose=False, device=device, conf=conf)[0]
    names = classes if is_world else yolo.names
    boxes = []
    for b in res.boxes:
        idx = int(b.cls)
        label = names[idx] if isinstance(names, dict) else names[idx]
        if label not in classes:          # closed-vocab: keep only requested classes
            continue
        x0, y0, x1, y1 = b.xyxy[0].tolist()
        boxes.append({"label": label, "box": (x0, y0, x1, y1)})
    return boxes


def sanitize(name):
    return re.sub(r"\s+", "_", name)


def write_results(out_dir, stem, meta, passes):
    json_path = out_dir / f"{stem}_results.json"
    lat = [p["latency_s"] for p in passes]
    summary = {
        "n_passes": len(passes),
        "n_passes_with_detections": sum(1 for p in passes if p["detections"]),
        "n_detections": sum(len(p["detections"]) for p in passes),
        "avg_latency_s": round(sum(lat) / len(lat), 4) if lat else None,
    }
    json_path.write_text(json.dumps({"meta": meta, "summary": summary, "passes": passes}, indent=2))

    csv_path = out_dir / f"{stem}_results.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["pass", "frame", "timestamp_s", "latency_s", "label",
                     "x0", "y0", "x1", "y1", "cx", "cy"])
        for p in passes:
            if not p["detections"]:
                wr.writerow([p["pass"], p["frame"], p["timestamp_s"], p["latency_s"],
                             "", "", "", "", "", "", ""])
            for d in p["detections"]:
                x0, y0, x1, y1 = d["box"]
                cx, cy = d["centroid"]
                wr.writerow([p["pass"], p["frame"], p["timestamp_s"], p["latency_s"],
                             d["label"], x0, y0, x1, y1, cx, cy])
    return json_path, csv_path


def run_video(yolo, is_world, video_path, classes, every, out_dir, meta_base, conf, device):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps * every))
    stem = sanitize(Path(video_path).stem)
    out_path = out_dir / f"{stem}_annotated.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"{n_frames} frames @ {fps:.1f} fps, detecting every {step} frames (~{every}s)")

    boxes, passes = [], []
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            t0 = time.perf_counter()
            boxes = detect(yolo, is_world, frame, classes, device, conf)
            dt = time.perf_counter() - t0
            for d in boxes:
                x0, y0, x1, y1 = d["box"]
                d["box"] = [round(v, 1) for v in (x0, y0, x1, y1)]
                d["centroid"] = [round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1)]
            passes.append({
                "pass": len(passes), "frame": i, "timestamp_s": round(i / fps, 3),
                "latency_s": round(dt, 4), "detections": boxes,
            })
            labels = ", ".join(d["label"] for d in boxes) or "nothing"
            print(f"  t={i / fps:6.1f}s  {len(boxes)} box(es): {labels}  ({dt * 1000:.0f}ms)")
        for det_ in boxes:
            x0, y0, x1, y1 = map(int, det_["box"])
            cv2.rectangle(frame, (x0, y0), (x1, y1), (72, 68, 230), 3)
            cv2.putText(frame, det_["label"], (x0 + 4, max(24, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (72, 68, 230), 2)
        writer.write(frame)
    cap.release()
    writer.release()

    meta = {**meta_base, "video": str(video_path), "fps": round(fps, 2),
            "resolution": [w, h], "detect_every_s": every,
            "run_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json_path, csv_path = write_results(out_dir, stem, meta, passes)
    lat = [p["latency_s"] for p in passes]
    print(f"\n{len(passes)} passes, avg {sum(lat) / max(len(lat), 1) * 1000:.0f} ms per pass")
    print(f"annotated video -> {out_path}")
    print(f"results -> {json_path}\nresults -> {csv_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?")
    ap.add_argument("classes", nargs="+")
    ap.add_argument("--video", metavar="PATH")
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quant", choices=["none", "int8", "int4"], default="none")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--run-name")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.video and args.image:
        args.classes = [args.image] + args.classes
        args.image = None
    if not (args.video or args.image):
        ap.error("provide an image path or --video")

    device = pick_device()
    yolo, is_world, footprint, n_q = load_model(args.model, args.classes, args.quant, device)
    meta_base = {"model": args.model, "quant": args.quant,
                 "model_footprint_gb": footprint, "device": device,
                 "classes": args.classes, "quantized_layers": n_q,
                 "conf_threshold": args.conf}

    if args.out:
        out_dir = Path(args.out)
    else:
        tag = Path(args.model).stem
        run_name = args.run_name or (f"{tag}_fp16" if args.quant == "none"
                                     else f"{tag}_{args.quant}")
        out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        run_video(yolo, is_world, args.video, args.classes, args.every, out_dir,
                  meta_base, args.conf, device)
        return

    import cv2
    frame = cv2.imread(args.image)
    t0 = time.perf_counter()
    boxes = detect(yolo, is_world, frame, args.classes, device, args.conf)
    print(f"\n{len(boxes)} detection(s) in {(time.perf_counter() - t0) * 1000:.0f} ms")
    for d in boxes:
        x0, y0, x1, y1 = d["box"]
        print(f"  {d['label']:<20} ({x0:.0f}, {y0:.0f}) - ({x1:.0f}, {y1:.0f})")


if __name__ == "__main__":
    main()
