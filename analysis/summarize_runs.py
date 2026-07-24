"""Aggregate every run under runs/ into one comparison table.

Scans runs/*/<video>_results.json, uses the fp16 (quant == "none") run as the
reference, and reports each run's footprint / hit rate / latency plus its
agreement with the reference (box IoU, centroid drift, misses). Writes
runs/summary.md and runs/summary.csv.

Usage:
    python summarize_runs.py
    python summarize_runs.py --runs runs --ref fp16_baseline
"""

import argparse
import csv
import json
from pathlib import Path


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


def load_runs(runs_dir):
    runs = {}
    for jf in sorted(Path(runs_dir).glob("*/*_results.json")):
        data = json.loads(jf.read_text())
        runs[jf.parent.name] = data
    return runs


def agreement(ref, cand):
    ref_f = {p["frame"]: p for p in ref["passes"]}
    both, miss, extra = 0, 0, 0
    ious, dists = [], []
    for f, cp in ((p["frame"], p) for p in cand["passes"]):
        if f not in ref_f:
            continue
        rb, cb = top_box(ref_f[f]), top_box(cp)
        if rb and cb:
            both += 1
            ious.append(iou(rb["box"], cb["box"]))
            dists.append(((rb["centroid"][0] - cb["centroid"][0]) ** 2 +
                          (rb["centroid"][1] - cb["centroid"][1]) ** 2) ** 0.5)
        elif rb and not cb:
            miss += 1
        elif cb and not rb:
            extra += 1
    return {
        "both": both, "missed_vs_ref": miss, "extra_vs_ref": extra,
        "mean_iou": round(sum(ious) / len(ious), 3) if ious else None,
        "mean_drift_px": round(sum(dists) / len(dists), 1) if dists else None,
    }


def row_for(name, data, ref_name, ref_data):
    m, s = data["meta"], data["summary"]
    hit = s["n_passes_with_detections"] / s["n_passes"] if s["n_passes"] else 0
    row = {
        "run": name,
        "quant": m.get("quant", "none"),
        "footprint_gb": m.get("model_footprint_gb"),
        "hit_rate": round(hit, 3),
        "avg_latency_s": s["avg_latency_s"],
        "mean_iou_vs_ref": "-",
        "mean_drift_px_vs_ref": "-",
        "missed_vs_ref": "-",
    }
    # IoU/drift vs ref only make sense within the same model AND same classes
    # (comparing quant levels of one detector), not across models/tasks.
    same = (m.get("model") == ref_data["meta"].get("model")
            and m.get("classes") == ref_data["meta"].get("classes"))
    if name != ref_name and same:
        a = agreement(ref_data, data)
        row["mean_iou_vs_ref"] = a["mean_iou"]
        row["mean_drift_px_vs_ref"] = a["mean_drift_px"]
        row["missed_vs_ref"] = a["missed_vs_ref"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--ref", default=None, help="reference run folder (default: the quant=none run)")
    args = ap.parse_args()

    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit(f"No */*_results.json under {args.runs}")

    ref_name = args.ref or next(
        (n for n, d in runs.items() if d["meta"].get("quant", "none") == "none"),
        next(iter(runs)),
    )
    ref_data = runs[ref_name]

    # Order: reference first, then by footprint descending.
    order = [ref_name] + sorted(
        (n for n in runs if n != ref_name),
        key=lambda n: runs[n]["meta"].get("model_footprint_gb") or 0, reverse=True,
    )
    rows = [row_for(n, runs[n], ref_name, ref_data) for n in order]

    cols = ["run", "quant", "footprint_gb", "hit_rate", "avg_latency_s",
            "mean_iou_vs_ref", "mean_drift_px_vs_ref", "missed_vs_ref"]
    headers = {
        "run": "Run", "quant": "Quant", "footprint_gb": "Size (GB)",
        "hit_rate": "Hit rate", "avg_latency_s": "Latency (s)",
        "mean_iou_vs_ref": "IoU vs ref", "mean_drift_px_vs_ref": "Drift px",
        "missed_vs_ref": "Missed",
    }

    def cell(r, c):
        v = r[c]
        if v is None:
            return "-"
        if c == "hit_rate" and isinstance(v, float):
            return f"{v:.0%}"
        return str(v)

    widths = {c: max(len(headers[c]), max(len(cell(r, c)) for r in rows)) for c in cols}
    line = "  ".join(headers[c].ljust(widths[c]) for c in cols)
    print(f"Reference: {ref_name}  |  video: {Path(ref_data['meta']['video']).name}")
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(cell(r, c).ljust(widths[c]) for c in cols))

    # Markdown
    md = [f"# Compression results — reference `{ref_name}`",
          f"\nVideo: `{Path(ref_data['meta']['video']).name}`  ·  "
          f"device: {ref_data['meta']['device']}  ·  "
          f"classes: {', '.join(ref_data['meta']['classes'])}\n",
          "| " + " | ".join(headers[c] for c in cols) + " |",
          "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        md.append("| " + " | ".join(cell(r, c) for c in cols) + " |")
    md.append("\n*IoU / Drift / Missed are measured against the reference run on "
              "frame-matched top boxes. Weight-only quant shrinks footprint but does "
              "not speed up MPS inference (no int matmul kernels).*")
    (Path(args.runs) / "summary.md").write_text("\n".join(md) + "\n")

    with open(Path(args.runs) / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n-> {args.runs}/summary.md")
    print(f"-> {args.runs}/summary.csv")


if __name__ == "__main__":
    main()
