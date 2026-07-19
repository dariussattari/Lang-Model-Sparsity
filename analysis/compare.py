"""Compare two detection runs produced by detect.py --video.

Treats the FIRST run as the reference (usually the fp16 baseline) and reports
how the SECOND run (usually a quantized/compressed model) differs on the same
video: size, latency, detection hit rate, and per-frame box agreement (IoU +
centroid drift). Because both runs sample the same video at the same interval,
passes are matched by frame index.

Usage:
    python compare.py runs/fp16_baseline runs/int8_quanto
    python compare.py runs/fp16_baseline/foo_results.json runs/int8_quanto/foo_results.json
"""

import json
import sys
from pathlib import Path


def load(path):
    p = Path(path)
    if p.is_dir():
        hits = sorted(p.glob("*_results.json"))
        if not hits:
            sys.exit(f"No *_results.json in {p}")
        p = hits[0]
    return json.loads(p.read_text()), p


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def top_box(p):
    """Largest-area detection in a pass, or None."""
    if not p["detections"]:
        return None
    return max(p["detections"], key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))


def centroid_dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def summarize(name, data):
    m, s = data["meta"], data["summary"]
    hit = s["n_passes_with_detections"] / s["n_passes"] if s["n_passes"] else 0
    print(f"  {name}")
    print(f"    model      : {m['model']}  ({m.get('quant', 'none')})")
    print(f"    footprint  : {m.get('model_footprint_gb', '?')} GB")
    print(f"    device     : {m['device']}")
    print(f"    hit rate   : {hit:.0%}  ({s['n_passes_with_detections']}/{s['n_passes']} passes)")
    print(f"    avg latency: {s['avg_latency_s']} s")
    return hit


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ref, ref_path = load(sys.argv[1])
    cmp, cmp_path = load(sys.argv[2])

    print("=" * 68)
    print("REFERENCE")
    ref_hit = summarize(sys.argv[1], ref)
    print("CANDIDATE")
    cmp_hit = summarize(sys.argv[2], cmp)
    print("=" * 68)

    # Frame-aligned agreement.
    ref_by_frame = {p["frame"]: p for p in ref["passes"]}
    cmp_by_frame = {p["frame"]: p for p in cmp["passes"]}
    shared = sorted(set(ref_by_frame) & set(cmp_by_frame))

    both, ref_only, cmp_only, neither = 0, 0, 0, 0
    ious, dists = [], []
    for f in shared:
        rb, cb = top_box(ref_by_frame[f]), top_box(cmp_by_frame[f])
        if rb and cb:
            both += 1
            ious.append(iou(rb["box"], cb["box"]))
            dists.append(centroid_dist(rb["centroid"], cb["centroid"]))
        elif rb and not cb:
            ref_only += 1
        elif cb and not rb:
            cmp_only += 1
        else:
            neither += 1

    print("AGREEMENT (matched by frame, top box per pass)")
    print(f"    shared passes      : {len(shared)}")
    print(f"    both detected      : {both}")
    print(f"    only reference     : {ref_only}   (candidate missed these)")
    print(f"    only candidate     : {cmp_only}   (candidate found extra)")
    print(f"    neither            : {neither}")
    if ious:
        print(f"    mean IoU (both)    : {sum(ious) / len(ious):.3f}")
        print(f"    median IoU (both)  : {sorted(ious)[len(ious) // 2]:.3f}")
        print(f"    mean centroid drift: {sum(dists) / len(dists):.1f} px")
    print("=" * 68)

    # Deltas.
    rf = ref["meta"].get("model_footprint_gb")
    cf = cmp["meta"].get("model_footprint_gb")
    print("DELTAS (candidate vs reference)")
    if rf and cf:
        print(f"    size    : {cf} GB vs {rf} GB  ({(cf / rf - 1) * 100:+.0f}%)")
    rl, cl = ref["summary"]["avg_latency_s"], cmp["summary"]["avg_latency_s"]
    if rl and cl:
        print(f"    latency : {cl} s vs {rl} s  ({(cl / rl - 1) * 100:+.0f}%)")
    print(f"    hit rate: {cmp_hit:.0%} vs {ref_hit:.0%}  ({(cmp_hit - ref_hit) * 100:+.0f} pts)")
    print("=" * 68)

    out = cmp_path.parent / "comparison.json"
    report = {
        "reference": str(ref_path),
        "candidate": str(cmp_path),
        "reference_footprint_gb": rf,
        "candidate_footprint_gb": cf,
        "reference_avg_latency_s": rl,
        "candidate_avg_latency_s": cl,
        "reference_hit_rate": round(ref_hit, 3),
        "candidate_hit_rate": round(cmp_hit, 3),
        "shared_passes": len(shared),
        "both_detected": both,
        "only_reference": ref_only,
        "only_candidate": cmp_only,
        "mean_iou": round(sum(ious) / len(ious), 3) if ious else None,
        "mean_centroid_drift_px": round(sum(dists) / len(dists), 1) if dists else None,
    }
    out.write_text(json.dumps(report, indent=2))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
