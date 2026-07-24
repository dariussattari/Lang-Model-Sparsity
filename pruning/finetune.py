"""Prune -> recover (fine-tune) -> benchmark, in one process.

Structured pruning collapses the detector (0% hit) because it shifts activation
statistics; Torch-Pruning's workflow recovers accuracy with a short fine-tune.
We fine-tune on the pseudo-labeled deployment video (see make_pseudo_dataset.py),
keeping the *pruned* architecture — the trick is subclassing DetectionTrainer so
ultralytics doesn't rebuild the model from the original YAML.

    python finetune.py --ratio 0.5 --epochs 40
    python finetune.py --ratio 0.5 --epochs 40 --quant-after int8   # recover then quantize
"""
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

import bench
import prune_lib

BASE = "drone_yolov8x.pt"
VIDEO = "quadcopter.mov"
CLASSES = ["drone"]
DATA = "dataset/drone.yaml"


def pruned_trainer_for(pruned_model):
    """A DetectionTrainer that trains the already-pruned model as-is."""
    class PrunedTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            pruned_model.args = self.args      # loss/criterion needs hyp
            return pruned_model
    return PrunedTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--quant-after", choices=["none", "int8", "int4"], default="none")
    ap.add_argument("--from-weights", default=None,
                    help="skip training: load an already-recovered .pt, then quant+benchmark")
    ap.add_argument("--device", default=None, help="override device (mps/cpu)")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    dev = args.device or bench.pick_device()
    print(f"device={dev} ratio={args.ratio} epochs={args.epochs} from={args.from_weights}")

    if args.from_weights:
        # reuse a recovered model (C2f_v2 resolvable because prune_lib is imported)
        import prune_lib as _  # noqa: keep C2f_v2 importable for unpickling
        ckpt = torch.load(args.from_weights, map_location=dev, weights_only=False)
        y = YOLO(BASE)  # gives us a YOLO wrapper; swap in the recovered module
        y.model = ckpt["model"].to(dev).float().eval()
        y.model.names = ckpt.get("names", {0: "drone"})
        stats = ckpt.get("prune_stats", {"target_channel_ratio": args.ratio})
    else:
        y = YOLO(BASE)
        y.model.to(dev).eval()
        stats = prune_lib.prune_model(y, args.ratio, imgsz=args.imgsz, device=dev)
        print(f"pruned: params -{stats['params_reduction_pct']}%  MACs -{stats['macs_reduction_pct']}%")

        # ---- recover ----
        # the downloaded checkpoint carries a stale v8DetectionLoss pinned to CUDA;
        # drop it so ultralytics rebuilds the criterion on this machine's device.
        y.model.criterion = None
        Trainer = pruned_trainer_for(y.model)
        y.train(trainer=Trainer, data=DATA, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=dev, project=str(Path("runs/_train").resolve()),
                name=f"prune{int(round(args.ratio*100))}", exist_ok=True,
                optimizer="AdamW", lr0=1e-3, cache=False, workers=4, verbose=False,
                plots=False, val=True, amp=False)
        # after training, y.model is the recovered model (float32)
        y.model.to(dev).float().eval()

    tag = f"drone_yolov8x_prune{int(round(args.ratio*100))}_ft"
    n_q = 0
    if args.quant_after != "none":
        from optimum.quanto import quantize, freeze, qint8, qint4
        qt = {"int8": qint8, "int4": qint4}[args.quant_after]
        quantize(y.model, weights=qt); freeze(y.model)
        n_q = sum(1 for m in y.model.modules()
                  if type(m).__name__.startswith(("QConv", "QLinear")))
        tag += f"_{args.quant_after}"

    run_name = args.run_name or tag
    out_dir = Path("runs") / run_name
    footprint = bench.model_footprint_gb(y.model)

    # save recovered weights (reload needs `import prune_lib`)
    if args.quant_after == "none":
        wpath = out_dir / f"{run_name}.pt"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": y.model, "names": y.names, "prune_stats": stats}, wpath)
        stats["weights_mb"] = round(wpath.stat().st_size / 1024**2, 2)

    meta_base = {"model": BASE, "quant": args.quant_after,
                 "model_footprint_gb": footprint, "device": dev,
                 "classes": CLASSES, "quantized_layers": n_q,
                 "conf_threshold": args.conf,
                 "pruning": {**stats, "recovered": True, "epochs": args.epochs}}
    result = bench.run_video(y, VIDEO, CLASSES, args.every, out_dir,
                             meta_base, args.conf, dev)
    (out_dir / "run_stats.json").write_text(json.dumps({
        "run_name": run_name, "ratio": args.ratio, "quant": args.quant_after,
        "recovered": True, "epochs": args.epochs, "footprint_gb": footprint,
        "n_passes": result["n_passes"], "hit": result["hit"],
        "hit_rate": round(result["hit"] / max(result["n_passes"], 1), 3),
        "avg_latency_s": result["avg_latency_s"], "pruning": stats,
    }, indent=2))
    print(f"DONE {run_name}: hit {result['hit']}/{result['n_passes']}  "
          f"foot {footprint}GB  lat {result['avg_latency_s']}s")


if __name__ == "__main__":
    main()
