# locateanything — NVIDIA LocateAnything-3B runner

Second model in the model-agnostic benchmark. Writes the same results schema as
`paligemma/` (see `../analysis/RESULTS_SCHEMA.md`), so the shared `../analysis/`
scripts compare the two without changes.

## Why its own directory + venv

LocateAnything pins `transformers==4.57.1` and loads via `trust_remote_code=True`
(Qwen2.5-3B + MoonViT, "Parallel Box Decoding"). That conflicts with PaliGemma's
transformers 5.x, so each model keeps its own venv.

## Setup

```bash
cd locateanything
uv venv --python cpython-3.12-macos-aarch64-none .venv
source .venv/bin/activate
uv pip install -r requirements.txt
hf auth login          # gated: accept NVIDIA license at huggingface.co/nvidia/LocateAnything-3B
```

**Apple Silicon quirk — decord stub.** The remote code statically imports
`decord` and `lmdb`. `lmdb` installs normally; `decord` has no arm64 macOS wheel,
so create a stub so transformers' import check passes (inference never calls it):

```bash
SP=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$SP/decord" && printf '__version__="0.0.0-stub"\n' > "$SP/decord/__init__.py"
```

## Usage

```bash
export PATH="$PWD/.venv/bin:$PATH"     # ninja on PATH for int4
python detect.py frame.jpg quadcopter                       # single image
python detect.py --video clip.mov quadcopter --quant none   # -> runs/fp16_baseline/
python detect.py --video clip.mov quadcopter --quant int8   # -> runs/int8_quanto/
python detect.py --video clip.mov quadcopter --quant int4   # -> runs/int4_quanto/
```

## Behavior notes (differs from PaliGemma)

- Prompt: `Locate all the instances that matches the following description: <cls>.`
  One prompt **per class** so each returned box carries the right label.
- Output is a decoded STRING like
  `<ref>quadcopter</ref><box><x1><y1><x2><y2></box>...` with coords normalized to
  `[0,1000]`; the runner rescales to pixels.
- `model.generate()` here returns that string directly (not token ids) and needs
  `tokenizer=` and `use_cache=True`.
- The model **over-generates duplicate boxes** (MTP decoding artifact); the runner
  caps `max_new_tokens=64` and dedupes identical coordinate tuples.
- ~4× slower per pass than PaliGemma at fp16 (bigger model + per-class prompting).
