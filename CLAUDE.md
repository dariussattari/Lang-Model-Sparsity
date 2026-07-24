# LM_Compression

Research project on **task-aware compression of LLMs and VLMs for edge deployment**,
using latent-space interpretability tools (sparse autoencoders and block-sparse
featurizers) to guide *what* gets compressed rather than compressing the whole
model uniformly.

## Core idea

Standard compression (quantization, pruning) treats every weight the same: GPTQ/AWQ
quantize all layers, Wanda prunes by magnitude×activation. None of them know *which
capability* a weight serves. But an edge deployment rarely needs a general-purpose
model — it needs one narrow task done well (e.g. "detect these object classes in
this camera's setting").

The thesis: **use encoders (SAEs / BSFs) as a scalpel, not just a microscope.**
Interpretability tools can identify the internal features and circuits a specific
task actually uses, so you can:

1. **Protect the task circuit, compress everything else.** Quantize/prune
   aggressively *outside* the features that carry the task, the way AWQ protects
   activation-salient channels — but with *semantic* salience (what the task needs)
   instead of purely statistical salience. See `Lit_Review/` #5 (Sparse Feature
   Circuits) and #24 (AWQ).
2. **Diagnose compression damage that perplexity misses.** Compressed models can
   look fine on aggregate loss while specific features are silently destroyed
   (Lit_Review #8). SAE/BSF feature-damage reports catch this and become the eval
   backbone.
3. **Repair with steering, not retraining.** After aggressive compression, small
   steering vectors (kilobytes) can push damaged representations back toward the
   dense model's, as a cheap alternative to LoRA recovery (Lit_Review #9–13, #27).

Vision is the priority and the thinnest part of the SAE ecosystem, so
**block-sparse featurizers** (Lit_Review #40, Goodfire) are the preferred encoder
for VLM vision towers — they model concepts as low-dimensional manifolds and ship
with vision-native reference code.

## Why "tactful" quantization beats whole-model quantization

- Whole-model INT4 gives a fixed size/quality tradeoff you can't steer.
- Task-aware compression lets you spend the bit budget where the task lives: keep
  higher precision on the ~task circuit, push the rest much lower, and (ideally)
  land at a smaller total footprint *at equal task accuracy* than uniform quant.
- The claim is only meaningful against a control. **Every result must be compared
  to standard compression (AWQ/GPTQ/Wanda) at a matched size budget**, measured on
  the task — not on perplexity.

## Recommended research substrate

- **Language backbone:** Gemma-2-2B / PaliGemma 2 — because **Gemma Scope** provides
  free pretrained SAEs at every layer (Lit_Review #3), which removes the single
  biggest hidden cost (training your own SAEs).
- **Vision tower:** train a small block-sparse featurizer (Lit_Review #40) on the
  VLM's own vision-tower activations — ideally on frames from the actual deployment
  distribution, which is narrow and should compress well.

## Repo layout (model-agnostic)

- `Lit_Review/` — 41 annotated papers + `00_Lit_Review_and_Approach.md`. Read first.
- `analysis/` — **shared, stdlib-only** analysis that works on ANY model's results:
  `compare.py` (two runs), `summarize_runs.py` (one model), `cross_model_report.py`
  (deltas between models). `RESULTS_SCHEMA.md` documents the JSON contract that makes
  this model-agnostic.
- `paligemma/` — VLM dir: `detect.py` runner + own venv (transformers 5.x) + `runs/`.
- `locateanything/` — VLM dir: `detect.py` runner + own venv (transformers 4.57 +
  `trust_remote_code`) + `runs/`.
- `yolo/` — CNN dir (YOLO / YOLO-World): `detect.py` + own venv (ultralytics) +
  `runs/`. Substrate for the CNN research track (torch.prune / GradCAM / calibrated
  PTQ instead of SAEs).
- Top-level `README.md` — the design and how to reproduce.

The design deliberately separates a model-neutral analysis layer from per-model
runners. Each model gets its own directory and venv (the two models pin
incompatible transformers versions) but writes the same results-JSON schema, so
`analysis/` compares them without knowing which model produced them. Adding a
third model = new dir + runner emitting the schema; analysis needs no changes.

## The eval harness

Each model's `detect.py` takes an image/webcam/video + class names and writes the
shared results schema. PaliGemma prompts `detect <class> ; <class>` and parses
`<locXXXX>` tokens; LocateAnything prompts "Locate all instances…" and parses
`<box>` tokens (one prompt per class so each box is labeled).

```bash
cd paligemma                                             # or locateanything
uv venv --python cpython-3.12-macos-aarch64-none .venv   # native arm64 (see below)
source .venv/bin/activate
uv pip install -r requirements.txt
hf auth login                                            # gated model; accept license first

python detect.py photo.jpg person "coffee mug"           # single image
python detect.py --camera person package                 # one webcam frame
python detect.py --video clip.mov quadcopter             # annotate a video + write results
```

Each `--video` run writes to its own subfolder under `runs/` (named from
`--quant`, or `--run-name NAME`) so baselines never overwrite each other:

```
runs/
  fp16_baseline/   <video>_annotated.mp4  <video>_results.{json,csv}
  int8_quanto/     ...
```

Artifacts per run:
- `<video>_annotated.mp4` — boxes drawn, persisting between detection passes.
- `<video>_results.json` — run metadata (model, **quant**, **model_footprint_gb**,
  device, classes, fps, resolution, sampling interval, UTC start), a summary
  block, and per-pass records with `timestamp_s`, `latency_s`, and each
  detection's `label`, `box [x0,y0,x1,y1]`, and `centroid [cx,cy]`.
- `<video>_results.csv` — same data flattened, one row per detection (loads into
  pandas). Passes with no detection still get a row so misses/latency stay visible.

**These results files are the comparison artifact.** The `analysis/` scripts diff
runs (from repo root):

```bash
python analysis/compare.py paligemma/runs/fp16_baseline paligemma/runs/int8_quanto
python analysis/summarize_runs.py --runs paligemma/runs
python analysis/cross_model_report.py paligemma/runs locateanything/runs
```

`compare.py` reports footprint, latency, hit rate, and — for passes matched by
frame index — box IoU and centroid drift; `cross_model_report.py` tabulates the
same across models at each quant level.

### Quantization baselines

`detect.py --quant {int8,int4}` applies **uniform whole-model** weight-only
quantization via `optimum-quanto` (portable to MPS, unlike CUDA-only
bitsandbytes/GPTQ/AWQ). This is the *control* the task-aware compression research
must beat: it shrinks footprint uniformly with no notion of which weights the task
needs. Weight-only quant on MPS shrinks memory but does **not** speed up inference
(no int matmul kernels — weights dequantize to fp16 per matmul); real latency wins
come from the target NPU. LM head left full-precision. int4 JIT-compiles a quanto
unpack kernel, so `ninja` must be on PATH.

### Measured baselines (quadcopter video, MPS, 34 passes)

PaliGemma-2-3B-mix-448:
| Quant | Size (GB) | Hit rate | Latency (s) | IoU vs fp16 |
| --- | --- | --- | --- | --- |
| fp16 | 5.65 | 94% | 1.13 | — |
| int8 | 3.38 | 94% | 3.03 | 0.973 |
| int4 | 3.446 | 91% | 13.4 | 0.833 |

LocateAnything-3B: fp16 = 7.115 GB, ~4.3 s/pass (int8/int4 rows land in
`analysis/cross_model.md`).

YOLO (CNN, `yolo/runs/summary.md`): yolo11n fp16 = 9.8 MB, 94% on *person* (COCO
has no drone class), **41 ms/pass** — ~30× faster than the VLMs. quanto int8/int4
shrinks it ~74% but detection **collapses to 0%** — naive weight-only quant
destroys compact CNNs (they need calibrated PTQ/QAT). YOLO-World-s (open-vocab)
gets only ~6% on the quadcopter — small CNNs can't do this drone; the VLMs can.

Takeaways that shape the research: (1) **int8 is the sweet spot** — ~40% smaller,
zero accuracy loss, boxes nearly identical (IoU 0.97). (2) **int4 via quanto is
strictly worse here** — not smaller than int8 (4-bit group-scale overhead), less
accurate, ~10× slower on MPS. (3) latency *rises* under quant on MPS — expected,
see above. (4) known weakness to track: box localization is loose and jitters
between passes; compression should not make this worse.

## Environment gotchas

- **Default `python3` on this machine is x86_64 (Rosetta)** via miniconda base, so
  `pip install torch` fails and MPS is unavailable. Always build ML venvs with
  `uv venv --python cpython-3.12-macos-aarch64-none`. Verify with
  `python -c "import platform; print(platform.machine())"` → must print `arm64`.
- Both models are **gated** — accept each license on Hugging Face while logged in,
  then `hf auth login` with a read token before first use.
- transformers 5.x renamed `torch_dtype` → `dtype` in `from_pretrained`.
- **quanto int4** needs `ninja` on PATH (JIT kernel) AND `torch.no_grad()` rather
  than `inference_mode()` (dequant sets version_counters).
- **LocateAnything** pins `transformers==4.57.1`, needs `trust_remote_code=True`,
  and its remote code statically imports `decord`+`lmdb`. `lmdb` installs fine;
  `decord` has no arm64 macOS wheel, so a stub package satisfies the import check
  (single-image inference never calls it). Its `generate()` returns a decoded
  STRING and requires `tokenizer=` + `use_cache=True`; it over-generates duplicate
  boxes (dedupe them).

## Roadmap (see Lit_Review/00 for detail)

0. Baselines — off-the-shelf VLM detection + eval harness (done).
1. Build a labeled eval set from real deployment footage.
2. SAE/BSF feature-damage diagnostics on a quantized model (Gemma Scope).
3. SAE/BSF-guided task pruning vs Wanda/Minitron at matched budget.
4. Steering-based recovery vs Recover-LoRA at matched byte budget.
5. Integrate the compressed model as the on-device slow path.
