"""Prune the drone YOLOv8x and benchmark it against the unpruned base.

Loads the drone-trained YOLOv8x (single class: 'drone'), optionally structurally
prunes it (Torch-Pruning) and/or quantizes it (optimum-quanto), then runs the
quadcopter video writing the shared results schema into runs/<name>/ so
../analysis/ can diff it against the existing yolo/ quantization baselines.

Examples
--------
    # baseline (no prune) + three pruning ratios, all benchmarked
    python run_experiment.py --ratios 0 0.25 0.5 0.75

    # a pruned model, then quantize it too (prune + quant combo)
    python run_experiment.py --ratios 0.5 --quant-after int8
"""
import argparse
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

import bench
import prune_lib

BASE_WEIGHTS = "drone_yolov8x.pt"
VIDEO = "quadcopter.mov"
CLASSES = ["drone"]


def apply_quant(model, quant, device):
    from optimum.quanto import quantize, freeze, qint8, qint4
    qtype = {"int8": qint8, "int4": qint4}[quant]
    print(f"Quantizing pruned model to {quant} via quanto ...", flush=True)
    quantize(model, weights=qtype)
    freeze(model)
    n = sum(1 for m in model.modules()
            if type(m).__name__.startswith(("QConv", "QLinear")))
    print(f"  {n} layers became quanto tensors", flush=True)
    return n


def run_one(ratio, quant, args, device):
    tag = "drone_yolov8x"
    if ratio and ratio > 0:
        tag += f"_prune{int(round(ratio * 100))}"
    if quant != "none":
        tag += f"_{quant}"
    if ratio in (0, None) and quant == "none":
        tag += "_fp16"
    run_name = args.run_name or tag
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 70}\nRUN: {run_name}  (ratio={ratio}, quant={quant})\n{'=' * 70}")

    yolo = YOLO(BASE_WEIGHTS)
    yolo.model.to(device).eval()

    prune_stats = None
    if ratio and ratio > 0:
        prune_stats = prune_lib.prune_model(yolo, ratio, imgsz=args.imgsz, device=device)
        print(f"  params {prune_stats['params_before']:,} -> {prune_stats['params_after']:,} "
              f"(-{prune_stats['params_reduction_pct']}%)  |  "
              f"MACs -{prune_stats['macs_reduction_pct']}%")
        # persist the pruned weights (reload needs `import prune_lib` for C2f_v2)
        wpath = out_dir / f"{run_name}.pt"
        torch.save({"model": yolo.model.half() if device == "cpu" else yolo.model,
                    "names": yolo.names, "prune_stats": prune_stats}, wpath)
        yolo.model.float().to(device)  # undo the transient half() for benchmarking
        prune_stats["weights_path"] = str(wpath)
        prune_stats["weights_mb"] = round(wpath.stat().st_size / 1024**2, 2)

    n_q = 0
    if quant != "none":
        n_q = apply_quant(yolo.model, quant, device)

    footprint = bench.model_footprint_gb(yolo.model)
    print(f"  footprint: {footprint} GB")

    meta_base = {
        "model": BASE_WEIGHTS, "quant": quant, "model_footprint_gb": footprint,
        "device": device, "classes": CLASSES, "quantized_layers": n_q,
        "conf_threshold": args.conf,
        "pruning": prune_stats or {"target_channel_ratio": 0.0},
    }
    result = bench.run_video(yolo, VIDEO, CLASSES, args.every, out_dir,
                             meta_base, args.conf, device)

    # a compact per-run stats file for the summary table
    (out_dir / "run_stats.json").write_text(json.dumps({
        "run_name": run_name, "ratio": ratio, "quant": quant,
        "footprint_gb": footprint, "n_passes": result["n_passes"],
        "hit": result["hit"], "hit_rate": round(result["hit"] / max(result["n_passes"], 1), 3),
        "avg_latency_s": result["avg_latency_s"], "pruning": prune_stats,
    }, indent=2))
    return run_name


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75],
                    help="channel pruning ratios to run (0 = unpruned baseline)")
    ap.add_argument("--quant-after", choices=["none", "int8", "int4"], default="none",
                    help="quantize each pruned model too (prune+quant combo)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--run-name", default=None, help="override (only sensible with one ratio)")
    args = ap.parse_args()

    device = bench.pick_device()
    print(f"device: {device}  |  base: {BASE_WEIGHTS}  |  video: {VIDEO}")
    t0 = time.time()
    done = []
    for ratio in args.ratios:
        done.append(run_one(ratio, args.quant_after, args, device))
    print(f"\nAll runs done in {time.time() - t0:.0f}s: {done}")


if __name__ == "__main__":
    main()
