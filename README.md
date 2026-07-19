# LM_Compression

Task-aware compression of detection VLMs for edge deployment, plus a
**model-agnostic** benchmark for measuring what compression costs. See
[CLAUDE.md](CLAUDE.md) for the research thesis and [Lit_Review/](Lit_Review/)
for the annotated literature.

## Model-agnostic design

The benchmark is deliberately split so the analysis never depends on one model:

```
LM_Compression/
  analysis/          # shared, stdlib-only — works on ANY model's results
    compare.py           # diff two runs (size / latency / hit-rate / IoU / drift)
    summarize_runs.py    # one model's runs -> table (per quant level)
    cross_model_report.py# deltas BETWEEN models, per quant level
  paligemma/         # model dir: runner + own venv (transformers 5.x) + runs/
  locateanything/    # model dir: runner + own venv (transformers 4.57 + remote code)
  Lit_Review/
```

The contract that makes it model-agnostic is the **results-JSON schema** every
runner emits (see [analysis/RESULTS_SCHEMA.md](analysis/RESULTS_SCHEMA.md)). Each
model gets its own directory with its own `detect.py` and its own venv (the two
models pin incompatible `transformers` versions), but they all write the same
schema, so the `analysis/` scripts — which import nothing heavier than the
standard library — compare them without knowing or caring which model produced
them. Adding a third model = a new directory + runner that writes the schema;
the analysis layer needs no changes.

## Models covered

| Dir | Model | Notes |
| --- | --- | --- |
| `paligemma/` | google/paligemma2-3b-mix-448 | native `detect` prompt, `<loc>` tokens |
| `locateanything/` | nvidia/LocateAnything-3B | Qwen2.5+MoonViT, `trust_remote_code`, `<box>` tokens |

Both are gated — accept each model's license on Hugging Face and `hf auth login`
before first use.

## Reproduce

Each model directory runs the same three-point sweep on the same video:

```bash
cd paligemma        # or locateanything
source .venv/bin/activate
# ninja must be on PATH for quanto int4's unpack kernel:
export PATH="$PWD/.venv/bin:$PATH"
python detect.py --video <clip.mov> quadcopter --quant none   # fp16 baseline
python detect.py --video <clip.mov> quadcopter --quant int8
python detect.py --video <clip.mov> quadcopter --quant int4
```

Then analyze (from repo root):

```bash
python analysis/summarize_runs.py --runs paligemma/runs        # one model
python analysis/compare.py paligemma/runs/fp16_baseline paligemma/runs/int8_quanto
python analysis/cross_model_report.py paligemma/runs locateanything/runs
```

Outputs land in `analysis/` (`cross_model.md`, `cross_model.csv`) and each
model's `runs/summary.{md,csv}`.

## The key caveat (read before trusting the latency column)

These runs are on Apple-Silicon **MPS**. `optimum-quanto` int8/int4 is
*weight-only* and portable, so it shrinks the model's memory footprint but does
**not** speed up MPS inference — there are no integer matmul kernels there, so
weights dequantize to fp16 before every matmul. Expect **footprint down, latency
flat-to-up**. Real latency wins come from the target NPU (e.g. Hailo), not this
laptop. The whole-model quant here is the *baseline* that task-aware compression
(the research thesis) aims to beat on the size/accuracy tradeoff.
