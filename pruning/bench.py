"""Shared benchmarking for the pruning experiment.

Writes the SAME results schema as ../yolo/detect.py and the VLM runners
(see ../analysis/RESULTS_SCHEMA.md), so ../analysis/ compares pruned runs against
the existing quantization baselines without changes.

Unlike yolo/detect.py this operates on an already-constructed ultralytics `YOLO`
object (the pruned model lives in-process), so there is no custom-module reload.
"""
import csv
import json
import sys
import time
from pathlib import Path

import torch


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_footprint_gb(module):
    """Real storage bytes (quant-aware), same walk as yolo/detect.py."""
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


def detect(yolo, image_bgr, classes, device, conf):
    """Return list of {label, box[x0,y0,x1,y1] px} for requested classes."""
    res = yolo.predict(image_bgr, verbose=False, device=device, conf=conf)[0]
    names = yolo.names
    boxes = []
    for b in res.boxes:
        label = names[int(b.cls)]
        if classes and label not in classes:
            continue
        x0, y0, x1, y1 = b.xyxy[0].tolist()
        boxes.append({"label": label, "box": (x0, y0, x1, y1)})
    return boxes


def _write_results(out_dir, stem, meta, passes):
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


def run_video(yolo, video_path, classes, every, out_dir, meta_base, conf, device,
              warmup=True):
    """Sample every `every` seconds, detect, draw boxes, write schema + annotated mp4."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps * every))
    stem = Path(video_path).stem.replace(" ", "_")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_annotated.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"{n_frames} frames @ {fps:.1f} fps, detecting every {step} frames (~{every}s)")

    # warm up so the first-pass latency isn't dominated by lazy MPS/kernel init
    if warmup:
        ok, f0 = cap.read()
        if ok:
            detect(yolo, f0, classes, device, conf)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    boxes, passes = [], []
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            t0 = time.perf_counter()
            boxes = detect(yolo, frame, classes, device, conf)
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
    json_path, csv_path = _write_results(out_dir, stem, meta, passes)
    lat = [p["latency_s"] for p in passes]
    hit = sum(1 for p in passes if p["detections"])
    print(f"\n{len(passes)} passes, {hit} with detections "
          f"({100 * hit / max(len(passes), 1):.0f}% hit), "
          f"avg {sum(lat) / max(len(lat), 1) * 1000:.0f} ms/pass")
    print(f"annotated -> {out_path}\nresults   -> {json_path}")
    return {"n_passes": len(passes), "hit": hit,
            "avg_latency_s": round(sum(lat) / max(len(lat), 1), 4)}
