"""Cross-model, cross-quant comparison table.

Model-agnostic: consumes the shared results-JSON schema written by any model's
detect.py runner. Give it one or more `runs/` directories (each containing
quant-named subfolders with *_results.json). The model label for each is the
name of the runs dir's PARENT folder (e.g. paligemma, locateanything).

Produces:
  - a per-(model, quant) table with size / hit-rate / latency and each row's
    delta vs that model's own fp16 baseline (intra-model deltas)
  - per-quant cross-model tables (deltas BETWEEN models at the same quant level)
  - optional cross-model detection agreement on frame-matched fp16 runs
  - writes <out>/cross_model.md and <out>/cross_model.csv

Usage:
    python cross_model_report.py paligemma/runs locateanything/runs
    python cross_model_report.py --out analysis paligemma/runs locateanything/runs
"""

import argparse
import csv
import json
from pathlib import Path


def load_model_runs(runs_dir):
    """Return {quant: data} for one model's runs dir; label from parent folder."""
    runs_dir = Path(runs_dir)
    label = runs_dir.parent.name or runs_dir.name
    out = {}
    for jf in sorted(runs_dir.glob("*/*_results.json")):
        data = json.loads(jf.read_text())
        out[data["meta"].get("quant", "none")] = data
    return label, out


def hit_rate(data):
    s = data["summary"]
    return s["n_passes_with_detections"] / s["n_passes"] if s["n_passes"] else 0.0


def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def top_box(p):
    if not p["detections"]:
        return None
    return max(p["detections"], key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))


def cross_agreement(a, b):
    """Frame-matched top-box agreement between two runs (possibly different models)."""
    af = {p["frame"]: p for p in a["passes"]}
    bf = {p["frame"]: p for p in b["passes"]}
    ious, dists, both = [], [], 0
    for f in set(af) & set(bf):
        ba, bb = top_box(af[f]), top_box(bf[f])
        if ba and bb:
            both += 1
            ious.append(iou(ba["box"], bb["box"]))
            dists.append(((ba["centroid"][0] - bb["centroid"][0]) ** 2 +
                          (ba["centroid"][1] - bb["centroid"][1]) ** 2) ** 0.5)
    return {
        "both": both,
        "mean_iou": round(sum(ious) / len(ious), 3) if ious else None,
        "mean_drift_px": round(sum(dists) / len(dists), 1) if dists else None,
    }


def fmt(v, pct=False):
    if v is None:
        return "-"
    if pct:
        return f"{v:.0%}"
    return str(v)


def pct_delta(cur, ref):
    if not cur or not ref:
        return "-"
    return f"{(cur / ref - 1) * 100:+.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dirs", nargs="+", help="one or more <model>/runs directories")
    ap.add_argument("--out", default="analysis", help="output directory for the report")
    args = ap.parse_args()

    models = {}
    order = []
    for rd in args.runs_dirs:
        label, runs = load_model_runs(rd)
        if not runs:
            print(f"warning: no results under {rd}")
            continue
        models[label] = runs
        order.append(label)

    if not models:
        raise SystemExit("no runs found")

    md, rows = [], []
    md.append("# Cross-model compression report\n")
    ref_video = next(iter(next(iter(models.values())).values()))["meta"]["video"]
    md.append(f"Video: `{Path(ref_video).name}`\n")

    # 1) Per-model table with intra-model deltas vs that model's fp16.
    md.append("## Per-model (deltas vs each model's own fp16)\n")
    md.append("| Model | Quant | Size (GB) | Hit rate | Latency (s) | Δ size | Δ latency |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    quant_order = ["none", "int8", "int4"]
    for label in order:
        runs = models[label]
        base = runs.get("none")
        bsize = base["meta"].get("model_footprint_gb") if base else None
        blat = base["summary"]["avg_latency_s"] if base else None
        for q in quant_order:
            if q not in runs:
                continue
            d = runs[q]
            size = d["meta"].get("model_footprint_gb")
            lat = d["summary"]["avg_latency_s"]
            hr = hit_rate(d)
            dsize = "-" if q == "none" else pct_delta(size, bsize)
            dlat = "-" if q == "none" else pct_delta(lat, blat)
            md.append(f"| {label} | {q} | {fmt(size)} | {fmt(hr, pct=True)} | "
                      f"{fmt(lat)} | {dsize} | {dlat} |")
            rows.append({"model": label, "quant": q, "size_gb": size,
                         "hit_rate": round(hr, 3), "latency_s": lat,
                         "delta_size_vs_fp16": dsize, "delta_latency_vs_fp16": dlat})

    # 2) Per-quant cross-model tables (model B vs model A at same quant).
    if len(order) >= 2:
        md.append("\n## Cross-model (same quant level, side by side)\n")
        a_label = order[0]
        for q in quant_order:
            present = [m for m in order if q in models[m]]
            if len(present) < 2:
                continue
            md.append(f"\n### quant = {q}\n")
            md.append("| Model | Size (GB) | Hit rate | Latency (s) | Δ size vs "
                      f"{a_label} | Δ latency vs {a_label} |")
            md.append("| --- | --- | --- | --- | --- | --- |")
            a = models[a_label][q] if q in models[a_label] else None
            asize = a["meta"].get("model_footprint_gb") if a else None
            alat = a["summary"]["avg_latency_s"] if a else None
            for m in present:
                d = models[m][q]
                size = d["meta"].get("model_footprint_gb")
                lat = d["summary"]["avg_latency_s"]
                ds = "-" if (m == a_label or not a) else pct_delta(size, asize)
                dl = "-" if (m == a_label or not a) else pct_delta(lat, alat)
                md.append(f"| {m} | {fmt(size)} | {fmt(hit_rate(d), pct=True)} | "
                          f"{fmt(lat)} | {ds} | {dl} |")

    # 3) Cross-model detection agreement on fp16 runs (do the models see the same thing?).
    if len(order) >= 2 and all("none" in models[m] for m in order[:2]):
        a, b = models[order[0]]["none"], models[order[1]]["none"]
        ag = cross_agreement(a, b)
        md.append(f"\n## Cross-model detection agreement ({order[0]} vs {order[1]}, fp16)\n")
        md.append(f"- frames both detected: {ag['both']}")
        md.append(f"- mean IoU of top boxes: {fmt(ag['mean_iou'])}")
        md.append(f"- mean centroid drift: {fmt(ag['mean_drift_px'])} px")
        md.append("\n*Low IoU here doesn't mean either model is wrong — the models "
                  "localize differently; it quantifies how interchangeable they are.*")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cross_model.md").write_text("\n".join(md) + "\n")
    with open(out / "cross_model.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["model", "quant", "size_gb", "hit_rate",
                                           "latency_s", "delta_size_vs_fp16",
                                           "delta_latency_vs_fp16"])
        wr.writeheader()
        wr.writerows(rows)

    print("\n".join(md))
    print(f"\n-> {out}/cross_model.md\n-> {out}/cross_model.csv")


if __name__ == "__main__":
    main()
