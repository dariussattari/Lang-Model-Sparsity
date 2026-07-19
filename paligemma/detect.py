"""PaliGemma 2 open-vocabulary detection demo.

Prompts a PaliGemma 2 "mix" checkpoint with `detect <class> ; <class> ...`
and parses the returned <locXXXX> tokens into bounding boxes.

Usage:
    python detect.py photo.jpg person dog "coffee mug"
    python detect.py --camera person package        # grab frames from webcam
    python detect.py --video clip.mov quadcopter    # annotate a video file
    python detect.py --video clip.mov quadcopter --quant int8   # quantized baseline
    python detect.py photo.jpg person --model google/paligemma2-3b-mix-224

Each --video run writes annotated video + results.{json,csv} into runs/<run-name>/.
Compare two runs with:  python compare.py runs/fp16_baseline runs/int8_quanto
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

DEFAULT_MODEL = "google/paligemma2-3b-mix-448"

# One detection = 4 location tokens (ymin, xmin, ymax, xmax on a 0-1023 grid)
# followed by the label, detections separated by ";".
LOC_RE = re.compile(
    r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*([^;<]+)"
)


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def model_footprint_gb(model):
    # Measure real storage bytes, deduped by underlying storage. quanto's
    # quantized tensors present as bf16 but keep int data in ._data/._scale,
    # so counting nelement*element_size overcounts them; walk storages instead.
    seen = {}
    for t in list(model.parameters()) + list(model.buffers()):
        sub = [getattr(t, a) for a in ("_data", "_scale", "_shift")
               if isinstance(getattr(t, a, None), torch.Tensor)]
        for s in (sub or [t]):
            try:
                st = s.untyped_storage()
                seen[st.data_ptr()] = st.nbytes()
            except Exception:
                seen[id(s)] = s.nelement() * s.element_size()
    return round(sum(seen.values()) / 1024**3, 3)


def load_model(model_id, quant="none"):
    device, dtype = pick_device()
    print(f"Loading {model_id} on {device} ({dtype}) ...", flush=True)
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype
    )

    if quant != "none":
        from optimum.quanto import quantize, freeze, qint8, qint4
        qtype = {"int8": qint8, "int4": qint4}[quant]
        print(f"Quantizing weights to {quant} (whole-model, uniform) ...", flush=True)
        # Exclude the LM head so token logits stay full precision (cheap, helps quality).
        quantize(model, weights=qtype, exclude="*lm_head*")
        freeze(model)

    model = model.to(device)
    model.eval()
    fp = model_footprint_gb(model)
    print(f"Model footprint: {fp} GB ({quant})", flush=True)
    return model, processor, device, fp


def detect(model, processor, device, image, classes):
    prompt = "<image>detect " + " ; ".join(classes)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(
        device, dtype=model.dtype
    )
    prompt_len = inputs["input_ids"].shape[-1]
    # no_grad (not inference_mode): quanto int4 dequant can't set version_counter
    # on inference tensors, which inference_mode would create.
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    decoded = processor.decode(out[0][prompt_len:], skip_special_tokens=False)

    w, h = image.size
    boxes = []
    for ymin, xmin, ymax, xmax, label in LOC_RE.findall(decoded):
        boxes.append(
            {
                "label": label.strip(),
                "box": (
                    int(xmin) / 1024 * w,
                    int(ymin) / 1024 * h,
                    int(xmax) / 1024 * w,
                    int(ymax) / 1024 * h,
                ),
            }
        )
    return boxes, decoded


def draw(image, boxes):
    out = image.copy()
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except OSError:
        font = ImageFont.load_default()
    colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0"]
    palette = {}
    for det in boxes:
        color = palette.setdefault(det["label"], colors[len(palette) % len(colors)])
        x0, y0, x1, y1 = det["box"]
        d.rectangle([x0, y0, x1, y1], outline=color, width=3)
        d.text((x0 + 4, max(0, y0 - 24)), det["label"], fill=color, font=font)
    return out


def grab_camera_frame():
    try:
        import cv2
    except ImportError:
        sys.exit("--camera needs opencv-python: pip install opencv-python")
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("Could not read from camera 0")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def write_results(out_dir, stem, meta, passes):
    """Write per-pass detection records as JSON (full) and CSV (flat)."""
    json_path = out_dir / f"{stem}_results.json"
    latencies = [p["latency_s"] for p in passes]
    summary = {
        "n_passes": len(passes),
        "n_passes_with_detections": sum(1 for p in passes if p["detections"]),
        "n_detections": sum(len(p["detections"]) for p in passes),
        "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
    }
    json_path.write_text(
        json.dumps({"meta": meta, "summary": summary, "passes": passes}, indent=2)
    )

    csv_path = out_dir / f"{stem}_results.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["pass", "frame", "timestamp_s", "latency_s", "label",
                     "x0", "y0", "x1", "y1", "cx", "cy"])
        for p in passes:
            if not p["detections"]:
                wr.writerow([p["pass"], p["frame"], p["timestamp_s"],
                             p["latency_s"], "", "", "", "", "", "", ""])
            for d in p["detections"]:
                x0, y0, x1, y1 = d["box"]
                cx, cy = d["centroid"]
                wr.writerow([p["pass"], p["frame"], p["timestamp_s"],
                             p["latency_s"], d["label"], x0, y0, x1, y1, cx, cy])
    return json_path, csv_path


def sanitize(name):
    return re.sub(r"\s+", "_", name)  # incl. the narrow no-break space in macOS filenames


def run_video(model, processor, device, video_path, classes, every, out_dir,
              model_id, quant="none", footprint_gb=None):
    try:
        import cv2
    except ImportError:
        sys.exit("--video needs opencv-python: pip install opencv-python")
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
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            t0 = time.perf_counter()
            boxes, _ = detect(model, processor, device, image, classes)
            dt = time.perf_counter() - t0
            for d in boxes:
                x0, y0, x1, y1 = d["box"]
                d["box"] = [round(v, 1) for v in (x0, y0, x1, y1)]
                d["centroid"] = [round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1)]
            passes.append({
                "pass": len(passes),
                "frame": i,
                "timestamp_s": round(i / fps, 3),
                "latency_s": round(dt, 3),
                "detections": boxes,
            })
            labels = ", ".join(d["label"] for d in boxes) or "nothing"
            print(f"  t={i / fps:6.1f}s  {len(boxes)} box(es): {labels}  ({dt:.1f}s)")
        # boxes persist until the next detection pass
        for det_ in boxes:
            x0, y0, x1, y1 = map(int, det_["box"])
            cv2.rectangle(frame, (x0, y0), (x1, y1), (72, 68, 230), 3)
            cv2.putText(frame, det_["label"], (x0 + 4, max(24, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (72, 68, 230), 2)
        writer.write(frame)
    cap.release()
    writer.release()

    meta = {
        "video": str(video_path),
        "model": model_id,
        "quant": quant,
        "model_footprint_gb": footprint_gb,
        "device": device,
        "classes": classes,
        "fps": round(fps, 2),
        "resolution": [w, h],
        "detect_every_s": every,
        "run_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json_path, csv_path = write_results(out_dir, stem, meta, passes)
    lat = [p["latency_s"] for p in passes]
    print(f"\n{len(passes)} detection passes, avg {sum(lat) / max(len(lat), 1):.1f}s per pass")
    print(f"annotated video -> {out_path}")
    print(f"results -> {json_path}")
    print(f"results -> {csv_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?", help="path to an image (omit with --camera)")
    ap.add_argument("classes", nargs="+", help="object classes to detect")
    ap.add_argument("--camera", action="store_true", help="capture one frame from webcam")
    ap.add_argument("--watch", type=float, metavar="SECS", default=0,
                    help="with --camera: re-detect every SECS seconds until Ctrl-C")
    ap.add_argument("--video", metavar="PATH", help="annotate a video file")
    ap.add_argument("--every", type=float, metavar="SECS", default=1.0,
                    help="with --video: seconds between detection passes (default 1.0)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quant", choices=["none", "int8", "int4"], default="none",
                    help="uniform whole-model weight quantization (optimum-quanto)")
    ap.add_argument("--run-name", metavar="NAME",
                    help="output subfolder under runs/ (default: derived from quant)")
    ap.add_argument("--out", default=None,
                    help="explicit output directory (overrides runs/<run-name>)")
    args = ap.parse_args()

    # argparse quirk: with --camera/--video the positional `image` slot eats the first class
    if (args.camera or args.video) and args.image:
        args.classes = [args.image] + args.classes
        args.image = None
    if not (args.camera or args.video or args.image):
        ap.error("provide an image path, --camera, or --video")

    model, processor, device, footprint = load_model(args.model, args.quant)

    if args.out:
        out_dir = Path(args.out)
    else:
        run_name = args.run_name or (
            "fp16_baseline" if args.quant == "none" else f"{args.quant}_quanto"
        )
        out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        run_video(model, processor, device, args.video, args.classes, args.every,
                  out_dir, args.model, args.quant, footprint)
        return

    def run_once(tag):
        image = grab_camera_frame() if args.camera else Image.open(args.image).convert("RGB")
        t0 = time.perf_counter()
        boxes, raw = detect(model, processor, device, image, args.classes)
        dt = time.perf_counter() - t0
        print(f"\n[{tag}] {len(boxes)} detection(s) in {dt:.1f}s  (raw: {raw.strip()[:120]})")
        for det in boxes:
            x0, y0, x1, y1 = det["box"]
            print(f"  {det['label']:<20} ({x0:.0f}, {y0:.0f}) - ({x1:.0f}, {y1:.0f})")
        path = out_dir / f"{tag}.jpg"
        draw(image, boxes).save(path)
        print(f"  annotated -> {path}")

    if args.camera and args.watch > 0:
        i = 0
        print(f"Watching camera every {args.watch}s, Ctrl-C to stop.")
        try:
            while True:
                run_once(f"frame_{i:04d}")
                i += 1
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nstopped.")
    else:
        run_once(Path(args.image).stem if args.image else "camera")


if __name__ == "__main__":
    main()
