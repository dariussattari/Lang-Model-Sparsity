"""NVIDIA LocateAnything-3B open-vocabulary detection runner.

Mirrors the PaliGemma demo's CLI and writes the *identical* results schema
(runs/<name>/<video>_results.{json,csv}) so the shared analysis scripts in
../analysis/ work model-agnostically across both models.

Differences from PaliGemma handled here:
  - loaded via AutoModel/AutoProcessor with trust_remote_code=True (Eagle family)
  - prompt: "Locate all the instances that matches the following description: ..."
  - output boxes are <box><x1><y1><x2><y2></box> as integers normalized to [0,1000]
  - one prompt per class, so each returned box carries the queried class label

Usage:
    python detect.py --video clip.mov quadcopter
    python detect.py --video clip.mov quadcopter --quant int8
    python detect.py photo.jpg person car
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
from transformers import AutoModel, AutoProcessor

DEFAULT_MODEL = "nvidia/LocateAnything-3B"

# <box><x1><y1><x2><y2></box> with coords normalized to [0, 1000].
BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
PROMPT = "Locate all the instances that matches the following description: {desc}."


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def model_footprint_gb(model):
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
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, dtype=dtype, attn_implementation="eager"
    )

    if quant != "none":
        from optimum.quanto import quantize, freeze, qint8, qint4
        qtype = {"int8": qint8, "int4": qint4}[quant]
        print(f"Quantizing weights to {quant} (whole-model, uniform) ...", flush=True)
        quantize(model, weights=qtype, exclude="*lm_head*")
        freeze(model)

    model = model.to(device)
    model.eval()
    fp = model_footprint_gb(model)
    print(f"Model footprint: {fp} GB ({quant})", flush=True)
    return model, processor, device, fp


# LocateAnything's custom generate() over-generates: it repeats the same box
# many times up to max_new_tokens. The distinct boxes appear in the first few,
# so we cap tokens and dedupe identical coordinate tuples.
MAX_NEW_TOKENS = 64


def detect_one_class(model, processor, device, image, cls):
    """Run the model for a single class; return list of deduped pixel boxes."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT.format(desc=cls)},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    # generate() returns the already-decoded answer STRING (not token ids), and
    # requires tokenizer + use_cache=True. no_grad (not inference_mode) so quanto
    # int4 dequant can set version_counters.
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             use_cache=True, tokenizer=processor.tokenizer)
    decoded = out if isinstance(out, str) else processor.decode(out[0], skip_special_tokens=False)

    w, h = image.size
    boxes, seen = [], set()
    for x1, y1, x2, y2 in BOX_RE.findall(decoded):
        key = (x1, y1, x2, y2)
        if key in seen:
            continue
        seen.add(key)
        boxes.append({
            "label": cls,
            "box": (int(x1) / 1000 * w, int(y1) / 1000 * h,
                    int(x2) / 1000 * w, int(y2) / 1000 * h),
        })
    return boxes


def detect(model, processor, device, image, classes):
    boxes = []
    for cls in classes:
        boxes.extend(detect_one_class(model, processor, device, image, cls))
    return boxes


def sanitize(name):
    return re.sub(r"\s+", "_", name)


def write_results(out_dir, stem, meta, passes):
    json_path = out_dir / f"{stem}_results.json"
    latencies = [p["latency_s"] for p in passes]
    summary = {
        "n_passes": len(passes),
        "n_passes_with_detections": sum(1 for p in passes if p["detections"]),
        "n_detections": sum(len(p["detections"]) for p in passes),
        "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
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


def run_video(model, processor, device, video_path, classes, every, out_dir,
              model_id, quant, footprint_gb):
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
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            t0 = time.perf_counter()
            boxes = detect(model, processor, device, image, classes)
            dt = time.perf_counter() - t0
            for d in boxes:
                x0, y0, x1, y1 = d["box"]
                d["box"] = [round(v, 1) for v in (x0, y0, x1, y1)]
                d["centroid"] = [round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1)]
            passes.append({
                "pass": len(passes), "frame": i, "timestamp_s": round(i / fps, 3),
                "latency_s": round(dt, 3), "detections": boxes,
            })
            labels = ", ".join(d["label"] for d in boxes) or "nothing"
            print(f"  t={i / fps:6.1f}s  {len(boxes)} box(es): {labels}  ({dt:.1f}s)")
        for det_ in boxes:
            x0, y0, x1, y1 = map(int, det_["box"])
            cv2.rectangle(frame, (x0, y0), (x1, y1), (72, 68, 230), 3)
            cv2.putText(frame, det_["label"], (x0 + 4, max(24, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (72, 68, 230), 2)
        writer.write(frame)
    cap.release()
    writer.release()

    meta = {
        "video": str(video_path), "model": model_id, "quant": quant,
        "model_footprint_gb": footprint_gb, "device": device, "classes": classes,
        "fps": round(fps, 2), "resolution": [w, h], "detect_every_s": every,
        "run_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json_path, csv_path = write_results(out_dir, stem, meta, passes)
    lat = [p["latency_s"] for p in passes]
    print(f"\n{len(passes)} passes, avg {sum(lat) / max(len(lat), 1):.1f}s per pass")
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
    ap.add_argument("--run-name")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.video and args.image:
        args.classes = [args.image] + args.classes
        args.image = None
    if not (args.video or args.image):
        ap.error("provide an image path or --video")

    model, processor, device, footprint = load_model(args.model, args.quant)

    if args.out:
        out_dir = Path(args.out)
    else:
        run_name = args.run_name or ("fp16_baseline" if args.quant == "none"
                                     else f"{args.quant}_quanto")
        out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        run_video(model, processor, device, args.video, args.classes, args.every,
                  out_dir, args.model, args.quant, footprint)
        return

    image = Image.open(args.image).convert("RGB")
    t0 = time.perf_counter()
    boxes = detect(model, processor, device, image, args.classes)
    print(f"\n{len(boxes)} detection(s) in {time.perf_counter() - t0:.1f}s")
    for d in boxes:
        x0, y0, x1, y1 = d["box"]
        print(f"  {d['label']:<20} ({x0:.0f}, {y0:.0f}) - ({x1:.0f}, {y1:.0f})")


if __name__ == "__main__":
    main()
