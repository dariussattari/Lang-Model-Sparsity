# initial_demo — can PaliGemma 2 detect our objects from a prompt?

Phase-0 sanity check for the compression research: PaliGemma 2 **mix** checkpoints
natively support detection via the prompt `detect class1 ; class2`, returning
`<locXXXX>` tokens (4 per box, ymin/xmin/ymax/xmax on a 0–1023 grid) plus a label.
If this model can find our objects, it's the substrate the SAE/BSF compression
work builds on (Gemma 2 backbone → free Gemma Scope SAEs).

## One-time setup

1. **Accept the license** (gated model): visit
   https://huggingface.co/google/paligemma2-3b-mix-448 while logged in and click accept.
2. **Authenticate + install:**
   ```bash
   cd initial_demo
   uv venv --python cpython-3.12-macos-aarch64-none .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   hf auth login        # paste a HF token with read access
   ```

   > ⚠️ Don't use the default `python3` on this machine — the miniconda base
   > Python is an **x86_64 (Rosetta)** build, which has no modern torch wheels
   > and no MPS GPU access. The `uv` command above pins a native arm64 Python.

First run downloads ~6 GB of weights. Runs on Apple Silicon (MPS) out of the box;
expect a few seconds per image on an M-series Mac, not real time — that's expected,
this is the "slow path" model.

## Usage

```bash
# single image, any classes you want (multi-word classes in quotes)
python detect.py photo.jpg person "coffee mug" laptop

# one frame from the webcam
python detect.py --camera person package

# webcam, re-detect every 5s until Ctrl-C (poor man's camera loop)
python detect.py --camera --watch 5 person package

# annotate a video: detection pass every second, boxes persist between passes,
# writes runs/<run-name>/<video>_annotated.mp4 + results.{json,csv}
python detect.py --video ~/Desktop/Screen*Recording*2026-07-10*.mov quadcopter
python detect.py --video clip.mov quadcopter --every 0.5   # denser passes

# whole-model (uniform) weight quantization baselines via optimum-quanto
python detect.py --video clip.mov quadcopter --quant int8  # -> runs/int8_quanto/
# int4 JIT-compiles a quanto unpack kernel -> ninja must be on PATH:
export PATH="$PWD/.venv/bin:$PATH"
python detect.py --video clip.mov quadcopter --quant int4  # -> runs/int4_quanto/ (slow on MPS)

# smaller/faster variant if memory is tight
python detect.py photo.jpg person --model google/paligemma2-3b-mix-224
```

Annotated images land in `detections/`; box coordinates print to the terminal.

### Results files (video mode)

Every `--video` run also writes structured results to `detections/`:

- **`<video>_results.json`** — run metadata (video, model, device, classes,
  fps, resolution, sampling interval, UTC start time), a summary block
  (passes, hit passes, total detections, avg latency), and one record per
  detection pass: `pass`, `frame`, `timestamp_s`, `latency_s`, and a
  `detections` list with `label`, `box` `[x0, y0, x1, y1]` (pixels), and
  `centroid` `[cx, cy]`.
- **`<video>_results.csv`** — the same data flattened to one row per
  detection (passes with no detections get a row with empty label/box
  columns so latency and misses stay visible). Loads directly into pandas:
  `pd.read_csv("detections/..._results.csv")`.

These files are the comparison artifact for the compression work: run the
same video through the uncompressed and compressed models and diff the
results files (hit rate, centroid drift, latency).

### Output layout

Each `--video` run writes to its own subfolder so baselines never overwrite
each other:

```
runs/
  fp16_baseline/   <video>_annotated.mp4  <video>_results.{json,csv}
  int8_quanto/     ...
  int4_quanto/     ...
```

Folder name comes from `--quant` (or override with `--run-name NAME`).

### Comparing two runs

```bash
# analysis scripts now live in ../analysis/ (shared, model-agnostic)
python ../analysis/compare.py runs/fp16_baseline runs/int8_quanto
python ../analysis/summarize_runs.py --runs runs
```

Treats the first run as reference and reports how the second differs on the
same video: model footprint (GB), avg latency, detection hit rate, and — for
passes matched by frame index — box **IoU** and **centroid drift** between the
two models' top detections. Writes `comparison.json` into the candidate folder.

> **Note on quantization + Apple Silicon:** `optimum-quanto` int8/int4 is
> *weight-only* and portable to MPS, so it shrinks the model's memory footprint
> but usually does **not** speed up inference on MPS — there are no integer
> matmul kernels there, so weights are dequantized to fp16 before each matmul.
> Expect footprint to drop and latency to stay flat or rise slightly. Real
> latency wins come from the target NPU (e.g. Hailo), not the laptop. This
> whole-model baseline is the control the task-aware compression work aims to
> beat on the size/accuracy tradeoff.

## What to look for

- **Does it find your classes at all?** Try the exact phrasings you'd use in
  deployment ("person", "delivery truck", "raccoon"). Phrasing matters —
  note which prompts work.
- **Failure modes**: small/far objects (try the 896-res checkpoint), unusual
  lighting, your specific camera angle.
- **Latency**: the printed seconds-per-frame is the uncompressed baseline number
  the compression work has to improve.

If detection quality is good enough here, next steps are: build the labeled
eval set from real camera footage, then run the standard-compression baselines
(AWQ/GPTQ/Wanda) on this same model and re-measure.
