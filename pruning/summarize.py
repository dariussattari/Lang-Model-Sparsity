"""Collate every run under runs/ into a comparison table + chart.

Reads runs/<run>/run_stats.json (written by run_experiment.py / finetune.py) plus
the run's results.json for the pass count, and emits:
    runs/summary.md, runs/summary.csv, runs/comparison.png
Ordering and labels are chosen to tell the pruning-vs-quant story.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("../analysis").resolve()))
import compare as C  # iou / top_box / centroid_dist, shared with the model-agnostic analysis

RUNS = Path("runs")


def iou_vs_base(run, base_run="drone_yolov8x_fp16"):
    """Frame-matched top-box IoU + centroid drift vs the fp16 base.

    Exposes 'phantom' detections (e.g. int8's fixed corner box) that inflate the
    naive hit rate: high hit rate but near-zero IoU == the boxes don't track the drone.
    """
    try:
        ref = json.loads((RUNS / base_run / "quadcopter_results.json").read_text())
        cand = json.loads((RUNS / run / "quadcopter_results.json").read_text())
    except FileNotFoundError:
        return None, None
    rp = {p["frame"]: p for p in ref["passes"]}
    ious, dists = [], []
    for p in cand["passes"]:
        rpp = rp.get(p["frame"])
        if not rpp or not rpp["detections"] or not p["detections"]:
            continue
        rb, cb = C.top_box(rpp), C.top_box(p)
        ious.append(C.iou(rb["box"], cb["box"]))
        dists.append(C.centroid_dist(rb["centroid"], cb["centroid"]))
    if not ious:
        return None, None
    return round(sum(ious) / len(ious), 3), round(sum(dists) / len(dists), 1)

# desired display order + human labels + method category (for colour)
ORDER = [
    ("drone_yolov8x_fp16",            "base (fp16, unpruned)",          "base"),
    ("drone_yolov8x_int8",            "quant int8 (no prune)",          "quant"),
    ("drone_yolov8x_prune25",         "prune 25%, no recovery",         "prune"),
    ("drone_yolov8x_prune50",         "prune 50%, no recovery",         "prune"),
    ("drone_yolov8x_prune75",         "prune 75%, no recovery",         "prune"),
    ("drone_yolov8x_prune50_ft",      "prune 50% + recovery (close)",   "prune_ft"),
    ("drone_yolov8x_prune75_ft",      "prune 75% + recovery (close)",   "prune_ft"),
    ("drone_yolov8x_prune50_ft_far",  "prune 50% + recovery (near+far)", "prune_far"),
    ("drone_yolov8x_prune75_ft_far",  "prune 75% + recovery (near+far)", "prune_far"),
    ("drone_yolov8x_prune50_ft_int8", "prune 50% + recovery + int8",    "combo"),
]


def load_runs():
    rows = []
    for run, label, cat in ORDER:
        sp = RUNS / run / "run_stats.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        pr = s.get("pruning") or {}
        miou, drift = (None, None) if cat == "base" else iou_vs_base(run)
        rows.append({
            "run": run, "label": label, "cat": cat,
            "footprint_gb": s["footprint_gb"],
            "hit": s["hit"], "n": s["n_passes"],
            "hit_rate": s["hit_rate"],
            "latency_ms": round(s["avg_latency_s"] * 1000, 1),
            "iou_vs_base": miou, "drift_px": drift,
            "params_reduction_pct": pr.get("params_reduction_pct", 0.0),
            "macs_reduction_pct": pr.get("macs_reduction_pct", 0.0),
        })
    return rows


def write_tables(rows):
    base = next((r for r in rows if r["cat"] == "base"), None)
    bfoot = base["footprint_gb"] if base else None
    lines = ["# Pruning experiment — drone YOLOv8x on the quadcopter video\n",
             "Base model: `doguilmak/Drone-Detection-YOLOv8x` (single class `drone`). "
             "Same video, sampling, and schema as the `yolo/` and VLM baselines. "
             "MPS, conf 0.25.\n",
             "| Run | Size (GB) | vs base | Params ↓ | Hit rate | IoU vs base | Drift px | Latency (ms) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        vs = f"{r['footprint_gb'] / bfoot * 100:.0f}%" if bfoot else "-"
        pr = f"-{r['params_reduction_pct']:.0f}%" if r["params_reduction_pct"] else "-"
        iou = "—" if r["iou_vs_base"] is None else f"{r['iou_vs_base']:.2f}"
        dr = "—" if r["drift_px"] is None else f"{r['drift_px']:.0f}"
        lines.append(f"| {r['label']} | {r['footprint_gb']:.4f} | {vs} | {pr} | "
                     f"{r['hit']}/{r['n']} ({r['hit_rate']*100:.0f}%) | {iou} | {dr} | {r['latency_ms']:.0f} |")
    lines += ["",
              "**Read the IoU column, not just hit rate.** Hit rate counts a pass as a hit if the "
              "model emits *any* box — so it is fooled by degenerate outputs. Naive **int8** scores "
              "a perfect hit rate yet ~0.26 IoU and >200 px centroid drift: it spams a fixed "
              "corner box on every frame and never tracks the drone (the project's own thesis — a "
              "metric looks fine while the capability is silently destroyed). "
              "No-recovery **pruning** collapses the other way, to 0 detections, because channel "
              "removal shifts the head's feature statistics. A short **recovery fine-tune** on the "
              "pseudo-labeled deployment video restores real, well-localized boxes (high IoU) at a "
              "fraction of the base size and latency."]
    (RUNS / "summary.md").write_text("\n".join(lines) + "\n")

    with open(RUNS / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "label", "footprint_gb", "hit", "n_passes", "hit_rate",
                    "iou_vs_base", "drift_px", "latency_ms", "params_reduction_pct",
                    "macs_reduction_pct"])
        for r in rows:
            w.writerow([r["run"], r["label"], r["footprint_gb"], r["hit"], r["n"],
                        r["hit_rate"], r["iou_vs_base"], r["drift_px"], r["latency_ms"],
                        r["params_reduction_pct"], r["macs_reduction_pct"]])


def make_chart(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"base": "#4c72b0", "quant": "#dd8452", "prune": "#c44e52",
              "prune_ft": "#55a868", "prune_far": "#2e7031", "combo": "#8172b3"}
    labels = [r["label"] for r in rows]
    y = range(len(rows))
    cols = [colors[r["cat"]] for r in rows]
    fig, ax = plt.subplots(1, 4, figsize=(18, 0.6 * len(rows) + 2), sharey=True)

    ax[0].barh(y, [r["hit_rate"] * 100 for r in rows], color=cols)
    ax[0].set_title("Hit rate (%)  — fooled by phantom boxes"); ax[0].set_xlim(0, 108)
    ax[1].barh(y, [(r["iou_vs_base"] or 0) for r in rows], color=cols)
    ax[1].set_title("IoU vs base  — real localization"); ax[1].set_xlim(0, 1.0)
    ax[2].barh(y, [r["footprint_gb"] for r in rows], color=cols)
    ax[2].set_title("Footprint (GB)")
    ax[3].barh(y, [r["latency_ms"] for r in rows], color=cols)
    ax[3].set_title("Latency (ms/frame, MPS)")
    for a in ax:
        a.invert_yaxis(); a.grid(axis="x", alpha=0.3)
    ax[0].set_yticks(list(y)); ax[0].set_yticklabels(labels, fontsize=9)
    for i, r in enumerate(rows):
        ax[0].text(r["hit_rate"]*100 + 1, i, f"{r['hit_rate']*100:.0f}", va="center", fontsize=8)
        iou = r["iou_vs_base"]
        ax[1].text((iou or 0) + 0.01, i, "—" if iou is None else f"{iou:.2f}", va="center", fontsize=8)
        ax[2].text(r["footprint_gb"], i, f" {r['footprint_gb']:.3f}", va="center", fontsize=8)
        ax[3].text(r["latency_ms"], i, f" {r['latency_ms']:.0f}", va="center", fontsize=8)
    fig.suptitle("Structured pruning vs quantization — drone YOLOv8x on the quadcopter video",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(RUNS / "comparison.png", dpi=130)
    print(f"wrote {RUNS/'comparison.png'}")


if __name__ == "__main__":
    rows = load_runs()
    write_tables(rows)
    make_chart(rows)
    print(f"summarized {len(rows)} runs -> runs/summary.md, summary.csv, comparison.png")
