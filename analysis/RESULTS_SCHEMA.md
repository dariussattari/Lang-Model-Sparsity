# Results-JSON schema (the model-agnostic contract)

Every model's `detect.py --video` writes `runs/<run>/<video>_results.json` in this
shape. The `analysis/` scripts depend ONLY on this schema — not on any model — so
any new model that emits it plugs into the same comparison tooling.

```jsonc
{
  "meta": {
    "video": "…/clip.mov",
    "model": "google/paligemma2-3b-mix-448",   // or nvidia/LocateAnything-3B, …
    "quant": "none",                            // "none" | "int8" | "int4"
    "model_footprint_gb": 5.65,                 // real storage bytes, quant-aware
    "device": "mps",
    "classes": ["quadcopter"],
    "fps": 24.87,
    "resolution": [506, 894],
    "detect_every_s": 1.0,
    "run_started_utc": "2026-07-11T00:23:32Z"
  },
  "summary": {
    "n_passes": 34,
    "n_passes_with_detections": 32,
    "n_detections": 32,
    "avg_latency_s": 1.132
  },
  "passes": [
    {
      "pass": 0,
      "frame": 0,
      "timestamp_s": 0.0,
      "latency_s": 1.511,
      "detections": [
        { "label": "quadcopter", "box": [x0, y0, x1, y1], "centroid": [cx, cy] }
      ]
    }
    // … one entry per sampled frame; empty "detections" for a miss
  ]
}
```

Rules a conforming runner must follow:
- `box` is `[x0, y0, x1, y1]` in **pixels** of the source frame (top-left origin).
- `centroid` is `[(x0+x1)/2, (y0+y1)/2]` in pixels.
- passes are sampled at a fixed `detect_every_s`, so runs on the same video are
  **frame-aligned** (matched by `frame`) — this is what lets compare/cross-model
  compute IoU and centroid drift between models or quant levels.
- `model_footprint_gb` must reflect real quantized storage (walk tensor storages;
  don't trust `element_size()` on quanto tensors, which present as bf16).
- a miss still produces a pass record (empty `detections`) so latency and
  miss-rate stay visible.

A CSV twin (`<video>_results.csv`, one row per detection) is written alongside for
pandas; the JSON is the source of truth.
