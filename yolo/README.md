# yolo — CNN detectors (YOLO family)

Third model family in the model-agnostic benchmark. Same results schema as the
VLMs (see `../analysis/RESULTS_SCHEMA.md`), so `../analysis/` compares them.
This is the substrate for the **CNN research track**: instead of SAEs, use
`torch.nn.utils.prune`, GradCAM, and calibrated PTQ/QAT.

## Setup

```bash
cd yolo
uv venv --python cpython-3.12-macos-aarch64-none .venv
source .venv/bin/activate
uv pip install -r requirements.txt        # ultralytics + quanto + CLIP (for YOLO-World)
export PATH="$PWD/.venv/bin:$PATH"         # ninja on PATH for quanto int4
```

Weights auto-download on first use (yolo11n ~5 MB, yolov8s-worldv2 ~340 MB).
No HF login needed — these are open.

## Usage

```bash
# stock COCO nano CNN (80 closed classes; detects person/car/... NOT drone)
python detect.py --video clip.mov person --model yolo11n.pt --quant none
python detect.py --video clip.mov person --model yolo11n.pt --quant int8
python detect.py --video clip.mov person --model yolo11n.pt --quant int4

# YOLO-World: open-vocab, takes a text prompt like the VLMs
python detect.py --video clip.mov quadcopter --model yolov8s-worldv2.pt
```

## Two model kinds (auto-detected from the weights name)

- **`*-world*`** → YOLO-World, **open-vocab**: `set_classes(<your classes>)`, needs
  the CLIP text encoder (installed from ultralytics' CLIP fork).
- **anything else** → stock **COCO** detector (closed 80-class vocab).
  `quadcopter`/`drone` are NOT COCO classes, so it scores 0 on the drone without
  fine-tuning — the runner prints a NOTE when a requested class isn't in the vocab.

## Findings on the quadcopter video (MPS, 34 passes)

| Run | Class | Size | Hit rate | Latency |
| --- | --- | --- | --- | --- |
| yolo11n fp16 | person | 9.8 MB | 94% | 41 ms |
| yolo11n int8 | person | 2.6 MB | **0%** | 32 ms |
| yolo11n int4 | person | 2.7 MB | **0%** | 187 ms |
| yolov8s-worldv2 fp16 | quadcopter | 611 MB | **6%** | 28 ms |

Three findings that shape the CNN track:

1. **CNNs are ~1000× faster than the VLMs** (28–41 ms vs 1–16 s per frame) and
   yolo11n is tiny (9.8 MB vs PaliGemma's 5.65 GB) — the real-time "fast path".
2. **Naive weight-only quant DESTROYS small CNNs.** quanto int8/int4 shrinks
   yolo11n ~74% (88 conv layers quantized) but detection collapses to 0% — small
   detectors have little redundancy, so they need **calibrated PTQ or QAT**
   (e.g. ultralytics `export(format=..., int8=True)` with a calibration set),
   not the naive per-tensor quant the big VLMs tolerated. This is the CNN analogue
   of the "compression damages specific capabilities" thesis.
3. **Open-vocab CNNs can't see this drone.** YOLO-World (even m/x) tops out at
   ~0.06 confidence on the quadcopter — 6% hit at conf 0.25 — where PaliGemma and
   LocateAnything both nail it. Detecting this drone needs a VLM or a
   drone-fine-tuned CNN.

## Next (the CNN research track, not done here)

- Calibrated int8 export (ultralytics) so quantized accuracy is actually usable.
- `torch` structured pruning of yolo11n + GradCAM to see which filters carry the
  task — the CNN parallel to SAE-guided task pruning.
- Fine-tune a nano CNN on drone footage so there's a real-time drone detector to
  compress (the fast path in the cascade).
