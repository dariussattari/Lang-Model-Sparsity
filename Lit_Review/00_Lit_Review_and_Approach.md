# Shrinking LLMs/VLMs for Edge Deployment via Latent-Space Methods
### Literature Review & Recommended Approach
*Last updated: 2026-07-09. All cited PDFs are in this folder, numbered to match the sections below.*

---

## 1. Framing: what the idea actually decomposes into

The proposal — "use SAEs to shrink models by exploiting the latent space, steering nodes/weights based on the input" — bundles together several distinct mechanisms. It's worth separating them, because they have very different maturity levels:

| Mechanism | What it does | Maturity | Key papers |
|---|---|---|---|
| **Quantization / pruning / distillation** | Actually shrinks bytes & FLOPs | Production-grade | GPTQ, AWQ, Wanda, Minitron (#18–28) |
| **Contextual (input-dependent) sparsity** | Activates only the weights a given input needs | Strong research systems (2–4× speedup) | Deja Vu, PowerInfer (#14–17) |
| **SAE-guided task pruning** | Uses interpretability to decide *what* to keep for a specific task | Emerging, publishable gap | Sparse Feature Circuits, SAE×compression papers (#5–8) |
| **Activation steering** | Controls behavior at inference; can recover/sharpen capabilities | Established for control, *not yet* for compression recovery | ActAdd, RepE, ITI (#9–13) |

The honest headline: **SAEs and steering do not, by themselves, make a model smaller.** An SAE is a *diagnostic and surgical instrument* — it tells you which internal features exist, which ones a task needs, and which ones compression damaged. Steering is a *control knob* — it moves behavior around without retraining. The bytes-on-disk shrinkage will always come from quantization + pruning + distillation. The genuinely novel research contribution available here is using the latent-space tools to make that shrinkage **task-aware**: you don't need a general-purpose model on the Pi, you need "detect deer and people in my backyard," and interpretability tools can identify and preserve exactly that circuitry while discarding the rest — something magnitude-based pruning (Wanda, SparseGPT) fundamentally cannot do, because it has no notion of *which capability* a weight serves.

Your "steer the nodes based on what the input is" intuition already has a validated ancestor: **contextual sparsity** (Deja Vu, #14). For any given input, a small input-*dependent* subset of attention heads and MLP neurons reproduces the dense model's output; a tiny predictor picks that subset on the fly. PowerInfer (#15) and LLM-in-a-flash (#16) turned this into real systems on constrained hardware. This is the strongest existing foundation for the input-adaptive half of your idea, and it composes with everything else.

## 2. Recommended architecture: a three-tier plan

### Tier 1 — The product path (works now): distill open-vocabulary detection into a compiled edge detector
For "detect objects of my choosing in a setting I choose," running a full VLM on every camera frame is the wrong tool — it's 100–1000× more compute than needed. The established recipe:

1. **Choose classes by text prompt** with an open-vocabulary detector — YOLO-World (#37) / YOLOE (#38) accept arbitrary class names and support *prompt-then-detect*: you bake your chosen vocabulary into the network offline, producing a compact closed-set detector with zero prompt overhead at inference.
2. **Optionally distill for your specific setting**: use a big open-vocab teacher (Grounding DINO #36, OWL-ViT #35) to auto-label footage from *your actual camera in your actual setting*, then train a small YOLO student on those labels. This is ViLD's insight (#34) — CLIP-level vocabulary knowledge distills into small detectors — plus free domain adaptation. TinyVLM (#39) pushes this to microcontroller scale with matryoshka embeddings, letting you keep a *degree* of open vocabulary in a tiny model.
3. **Quantize to INT8 and compile to a Hailo `.hef`**. YOLOv8-class models run at real-time FPS on the Hailo-8/8L with official support.

This tier gives you a working camera product early and becomes the *baseline your research must beat*.

### Tier 2 — The VLM path (your priority): small VLM as the "slow path" in a cascade
Small VLMs are now genuinely deployable: SmolVLM (#29, 256M–2.2B), Moondream (0.5B–2B, detection/pointing/counting built in), MobileVLM (#30), TinyLLaVA (#31), MiniCPM-V (#32). FastVLM (#33) matters specifically for camera work — it attacks vision-encoder latency, which dominates VLM time-to-first-token at high image resolution.

**Hardware reality check** (this changes your shopping list):
- The **original AI HAT+ (Hailo-8L 13 TOPS / Hailo-8 26 TOPS)** has no onboard DRAM and is built for vision CNNs. It runs the Tier-1 detector beautifully; it **cannot run an LLM/VLM**.
- The **AI HAT+ 2 (Hailo-10H, 40 TOPS INT4, 8 GB dedicated LPDDR4, ~$130, released Jan 2026)** is explicitly designed for on-device generative AI — Raspberry Pi's supported targets include **Qwen2.5-VL-3B**, Llama-3.2-3B, and DeepSeek-R1-Distill-1.5B. If VLMs are the priority, this is the board to buy. Fallback: 4-bit GGUF Moondream/SmolVLM on the Pi 5 CPU via llama.cpp runs at usable-for-slow-path speeds.

**The cascade architecture** I'd recommend for the camera:
- **Fast path** (Hailo, every frame, ~30 FPS): Tier-1 compiled detector for your chosen classes.
- **Slow path** (VLM, on trigger or every N seconds): scene-level reasoning the detector can't do — verification of low-confidence detections, relational queries ("is the person carrying a package?"), and *continual auto-labeling* to improve the fast path over time.

### Tier 3 — The research contribution: SAE-guided, task-aware compression with steering-based recovery
This is where the novel work lives. Four concrete, publishable directions, in order of feasibility:

**(a) SAE-guided task pruning.** Sparse Feature Circuits (#5) shows SAE features form causal circuits for specific behaviors, and that you can *ablate everything outside a circuit* (their SHIFT method) to surgically remove capabilities. Invert the logic: identify the circuit for *your* task (object grounding, spatial language, your class vocabulary), then prune/quantize aggressively everywhere *outside* it — protecting task-critical channels the way AWQ (#24) protects activation-salient channels, but with *semantic* rather than statistical salience. Baseline to beat: Wanda (#19) and Minitron-style prune+distill (#22) at equal parameter budget, evaluated on your task rather than perplexity.

**(b) SAEs as compression diagnostics.** Paper #8 (2026) shows **perplexity can miss SAE-feature damage under quantization** — models look fine on perplexity while specific features are silently destroyed. Paper #6 shows SAEs transfer to compressed models well enough to make this practical, and #7 does mechanistic analysis of compression in VLMs specifically. A "feature-damage report card" for compressed VLMs is a low-hanging, useful contribution and the evaluation backbone for (a) and (c).

**(c) Steering-based recovery of compressed models.** After aggressive compression, use activation steering (ActAdd #9, RepE #10, ITI #11, SAE-targeted steering #12) to push damaged representations back toward the dense model's — a few steering vectors cost kilobytes versus megabytes for LoRA recovery (Recover-LoRA, #27, is the LoRA-based comparison point). Nobody has systematically studied steering-to-repair-quantization-damage; #8 explicitly flags it as open. Caution: #13 documents when steering vectors fail unpredictably — build its reliability checks into your protocol.

**(d) Input-conditional execution on device.** Deja Vu-style predictors (#14) gating which heads/MLP blocks load, specialized to your deployment distribution (a fixed camera sees a *narrow* input distribution — contextual sparsity should be far higher than the general-case 2×). #17 (on-demand multi-task sparsity on edge devices) is the closest recent system.

**Practical model choice for Tier 3:** do the research on **Gemma-2-2B/Gemma-3**, because Gemma Scope (#3) provides free, high-quality pretrained SAEs at every layer — training your own SAEs is the single biggest hidden cost in this agenda (see #2 for what that takes). For the VLM side, SAEs on the vision tower/CLIP are an active but thinner area (#7 is the closest); expect to train small featurizers yourself there — and consider **block-sparse featurizers (#40)** instead of vanilla SAEs for the vision tower, since they were designed for vision models and come with open reference code.

## 3. Suggested roadmap

| Phase | Goal | Deliverable |
|---|---|---|
| 0 | Hardware + baselines | Pi 5 + AI HAT+ 2; YOLOv8n compiled to Hailo detecting your classes; Moondream/Qwen2.5-VL running quantized |
| 1 | Task-specific distillation loop | Teacher (Grounding DINO) auto-labels your camera footage → small student detector; measure vs off-the-shelf |
| 2 | SAE diagnostics | Feature-damage report for GPTQ/AWQ-quantized Gemma-2-2B using Gemma Scope; replicate #8's perplexity-blindness finding |
| 3 | SAE-guided task pruning | Circuit-protected pruning vs Wanda/Minitron at matched budget on task metrics |
| 4 | Steering recovery | Steering vectors to repair compression damage; compare vs Recover-LoRA at matched byte budget |
| 5 | Integration | Compressed, steered VLM as slow path on the HAT+ 2, cascaded with the Hailo fast-path detector |

---

## 4. Annotated bibliography

Filenames in this folder are `NN_Name_arxivID.pdf`.

### A. Sparse autoencoders — foundations (#1–4)
1. **Cunningham et al. 2023 — Sparse Autoencoders Find Highly Interpretable Features in Language Models** (2309.08600). The paper that started the SAE wave: dictionary learning on residual-stream activations yields far more monosemantic features than neurons. Read first for the core method.
2. **Gao et al. (OpenAI) 2024 — Scaling and evaluating sparse autoencoders** (2406.04093). TopK SAEs, scaling laws, and evaluation metrics. Your reference for what SAE training actually costs and how to judge SAE quality.
3. **Lieberum et al. (DeepMind) 2024 — Gemma Scope** (2408.05147). Open pretrained SAEs on every layer of Gemma 2. *The practical reason to build on Gemma:* skips the most expensive step of the whole agenda.
4. **Shu et al. 2025 — A Survey on Sparse Autoencoders** (2503.05613). Map of the field: architectures, training, applications, limitations.

### B. SAEs × compression — the core research gap (#5–8)
5. **Marks et al. 2024 — Sparse Feature Circuits** (2403.19647). Causal circuits of SAE features; ablating outside a circuit surgically removes capabilities. *The methodological backbone of the task-pruning proposal.*
6. **2025 — On the transferability of SAEs for interpreting compressed models** (2507.15977). SAEs trained on the original model still interpret its pruned/quantized versions; pruning the SAE itself works too. Makes SAE-based compression evaluation cheap.
7. **2026 — Mechanistically Interpreting Compression in VLMs** (2603.25035). Closest existing work to the VLM half of this project — what compression does to VLM internals.
8. **2026 — Perplexity Can Miss SAE Feature Damage Under Quantization** (2606.03002). Quantized models with near-identical perplexity can have badly damaged features. Justifies the SAE-diagnostics phase and flags steering-under-quantization as open.

### B2. Beyond SAEs — alternative featurizers (#40)
40. **Fel, Kowal et al. (Goodfire) 2026 — Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds** (2606.25234). Replaces the SAE's unit of sparsity (a single direction) with a *block* of directions, modeling each concept as a low-dimensional (2–4D) manifold — a sparse sum of manifolds rather than a sparse sum of atoms. Three variants (vanilla block-TopK, Grassmannian, group-lasso); recovers curve-detector manifolds in InceptionV1, finds novel shadow/lighting manifolds in DINOv3, and steers SDXL generation via manifold coordinates. *Directly relevant here for two reasons:* (1) it's built for **vision models** — the thinnest part of the SAE ecosystem and the half of the VLM this project cares most about; (2) block structure gives every downstream use in Tier 3 a better handle — task pruning protects whole concept manifolds instead of scattered atoms, and steering gets *coordinates within* a concept (e.g., where along the manifold) rather than a single on/off direction. Reference code: [goodfire-ai/block-sparse-featurizer](https://github.com/goodfire-ai/block-sparse-featurizer) (MIT, DINOv3 demo included). Caveat: interpretability/steering results only — no compression or efficiency benchmarks yet, so treat it as a candidate *featurizer swap* inside the Tier-3 pipeline, not evidence the pipeline works.

### C. Activation steering (#9–13)
9. **Turner et al. 2023 — Steering Language Models With Activation Engineering** (2308.10248). ActAdd: add activation-difference vectors at inference to steer behavior. Simplest steering method; start here.
10. **Zou et al. 2023 — Representation Engineering** (2310.01405). Reading vectors and control at the representation level, systematically across many behaviors.
11. **Li et al. 2023 — Inference-Time Intervention** (2306.03341). Steering along truthful directions in selected attention heads; the template for *targeted, head-level* intervention.
12. **2025 — Steering Target Atoms** (2505.20322). Steering through SAE-decomposed features rather than raw directions — the bridge between your SAE and steering components.
13. **2025 — Understanding (Un)Reliability of Steering Vectors** (2505.22637). When and why steering fails. Required reading before building anything load-bearing on steering.

### D. Input-conditional computation / contextual sparsity (#14–17)
14. **Liu et al. 2023 — Deja Vu: Contextual Sparsity** (2310.17157). Input-dependent subsets of heads/neurons ≈ dense output; lightweight predictors select them on the fly; ~2× speedup, no quality loss. *The validated form of "steer which nodes run based on the input."*
15. **Song et al. 2023 — PowerInfer** (2312.12456). Hot/cold neuron split across GPU/CPU using activation locality — contextual sparsity as a deployed system on consumer hardware.
16. **Alizadeh et al. (Apple) 2023 — LLM in a flash** (2312.11514). Runs models larger than DRAM by streaming only activated weights from flash — directly relevant to Pi-class memory budgets.
17. **2025 — On-Demand Multi-Task Sparsity for Edge Devices** (2511.19986). Recent edge-focused treatment; closest system to what Phase 5 would build.

### E. Pruning (#18–22)
18. **Frantar & Alistarh 2023 — SparseGPT** (2301.00774). One-shot 50% unstructured pruning of GPT-scale models.
19. **Sun et al. 2023 — Wanda** (2306.11695). Prune by |weight|×‖activation‖ — trivially simple, strong baseline. *Your primary pruning baseline.*
20. **Ma et al. 2023 — LLM-Pruner** (2305.11627). Gradient-based *structural* pruning (structural = real speedup on real hardware, which is what the Pi needs).
21. **Ashkboos et al. 2024 — SliceGPT** (2401.15024). Deletes whole rows/columns via orthogonal transforms — up to ~25% parameter reduction with dense-hardware-friendly results.
22. **Muralidharan et al. (NVIDIA) 2024 — Minitron** (2407.14679). Prune + distill to recover: 15T-token models compressed with ~40× fewer training tokens. *The industrial-strength baseline Tier-3(a) must beat.*

### F. Quantization (#23–27)
23. **Frantar et al. 2022 — GPTQ** (2210.17323). Second-order post-training quantization to 3–4 bit; foundational.
24. **Lin et al. 2023 — AWQ** (2306.00978). Protects the ~1% activation-salient channels; conceptually the closest existing method to "protect what matters, compress the rest" — Tier-3(a) replaces its statistical salience with SAE-derived semantic salience.
25. **Xiao et al. 2022 — SmoothQuant** (2211.10438). W8A8 by migrating activation outliers into weights; relevant because NPUs (incl. Hailo) want integer activations too.
26. **Ma et al. 2024 — BitNet b1.58** (2402.17764). Ternary-weight LLMs; where extreme low-bit is headed (note bitnet.cpp runs well on ARM CPUs like the Pi's).
27. **2026 — Recover-LoRA for Aggressive Quantization** (2606.04238). Recovering 2-bit models with LoRA + KD on synthetic data. *The comparison point for steering-based recovery, Tier-3(c).*

### G. Distillation (#28)
28. **Gu et al. 2023 — MiniLLM** (2306.08543). On-policy reverse-KLD distillation of generative LLMs; the modern distillation recipe if you distill your own small VLM/LLM.

### H. Small / edge VLMs (#29–33)
29. **Marafioti et al. (HF) 2025 — SmolVLM** (2504.05299). 256M–2.2B VLMs designed for small memory; strong architecture/tokenization lessons for tiny multimodal models.
30. **Chu et al. 2023 — MobileVLM** (2312.16886). Mobile-first VLM with efficient projector; latency-focused design.
31. **Zhou et al. 2024 — TinyLLaVA** (2402.14289). Framework + ablations for small multimodal training — what matters (data, recipe) vs what doesn't (sheer scale).
32. **Yao et al. 2024 — MiniCPM-V** (2408.01800). GPT-4V-class at 8B on phones; the strong-end reference for edge VLM capability.
33. **Vasu et al. (Apple) 2024 — FastVLM** (2412.13303). Hybrid vision encoder cutting TTFT dramatically at high resolution — the binding constraint for camera-frame VLM inference.

### I. Open-vocabulary detection for the camera (#34–39)
34. **Gu et al. 2021 — ViLD** (2104.13921). The original "distill CLIP into a detector" — the conceptual license for Tier 1's teacher→student loop.
35. **Minderer et al. 2022 — OWL-ViT** (2205.06230). Simple, reliable open-vocab detector; good auto-labeling teacher.
36. **Liu et al. 2023 — Grounding DINO** (2303.05499). Strongest text-conditioned detector; best auto-labeling teacher, too heavy for the Pi itself.
37. **Cheng et al. 2024 — YOLO-World** (2401.17270). Real-time open-vocab YOLO; *prompt-then-detect* re-parameterization bakes your vocabulary into a compact deployable model. Cornerstone of Tier 1.
38. **Wang et al. 2025 — YOLOE** (2503.07465). Successor: text/visual/prompt-free modes, ~3× cheaper training than YOLO-World, RepRTA re-parameterization with zero inference overhead.
39. **2026 — TinyVLM: Zero-Shot Detection on Microcontrollers** (2603.00136). Vision-language distillation with matryoshka embeddings at MCU scale — proof the open-vocab→tiny-edge path compresses further than the Pi even needs.
41. **NVIDIA 2026 — LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding** (2605.27365). A 3B grounding VLM (Qwen2.5-3B LLM + MoonViT-SO-400M vision, Eagle family) that takes natural-language queries and returns boxes/points directly — Parallel Box Decoding predicts each box as one structured unit instead of autoregressive coordinate tokens (~2.5× faster grounding). Trained on 12M images / 785M boxes across natural scenes, robotics, driving, GUI, and documents. *Why it matters here:* it collapses the detector-vs-VLM split — one model that both understands language and localizes — making it (a) the strongest current **teacher** for auto-labeling camera footage, (b) a candidate **slow-path model** replacing a generic VLM, and (c) arguably the ideal **Tier-3 compression target**: a brand-new, uncompressed, detect-anything model that a fixed camera only needs a sliver of. Caveats: NVIDIA Research License (non-commercial restrictions — fine for research, an issue for productizing); Qwen backbone means no free pretrained SAEs (unlike the PaliGemma 2 route); and 3B-VLM-per-query is still not real-time on Pi-class hardware, so it lives in the slow path or the lab, not the 30 FPS loop.

---

## 5. Key risks to keep in view
- **SAE features ≠ guaranteed causal units.** Protecting SAE-identified circuits during pruning may under-deliver if features are distributed redundantly; always keep the Wanda/Minitron ablation as the control.
- **Steering reliability** (#13): effects vary by prompt distribution; a fixed camera's narrow distribution actually *helps* here, but validate per-deployment.
- **Hailo compiler constraints**: the `.hef` toolchain supports a specific op set; exotic architectures (custom gating, dynamic sparsity) will fall back to CPU. Design Tier-3(d) around what the NPU can express, or accept CPU execution for the LLM side.
- **VLM SAEs are thin ground**: most SAE infrastructure targets text-only residual streams. Budget time for training vision-tower featurizers — block-sparse featurizers (#40) are the most promising vision-native option — or scope Tier 3 to the language backbone of the VLM first.
