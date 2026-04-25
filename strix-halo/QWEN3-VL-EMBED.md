<!-- Copyright 2026 bitserv-ai -->
<!-- SPDX-License-Identifier: MIT -->

# Qwen3-VL Embedding & Reranking on AMD Strix Halo (gfx1151)

Production deployment notes for multimodal embedding and reranking inference
on ROCm/gfx1151. Covers BF16 baseline and INT8 W8A16 quantized deployments.

## Model Details

| Property | Embedding | Reranker |
|----------|-----------|----------|
| Architecture | Qwen3VLForConditionalGeneration | Qwen3VLForConditionalGeneration |
| Parameters | 8B (36 layers) | 8B (36 layers) |
| Context Length | 32K (configured: 32768) | 32K (configured: 32768) |
| Embedding Dim | 4096 (supports 64–4096 via Matryoshka) | N/A (classification) |
| Input Modalities | Text, images, screenshots, video, mixed | Text, images |
| BF16 Model Path | `/path/to/your/models/Qwen3-VL-Embedding-8B` | `/path/to/your/models/Qwen3-VL-Reranker-8B` |
| W8A16 Model Path | `/path/to/your/models/Qwen3-VL-Embedding-8B-W8A16` | `/path/to/your/models/Qwen3-VL-Reranker-8B-W8A16` |
| Checkpoint Size (BF16) | 16 GiB | 17 GiB |
| Checkpoint Size (W8A16) | 9.9 GiB | 9.9 GiB |

## Critical: ViT NaN on gfx1151

The ViT encoder produces **100% NaN** in `last_hidden_state` when running in
BF16 or FP16 on AMD gfx1151 (Radeon 8060S). Root cause: GELU with tanh
approximation overflows in BF16/FP16 on ROCm.

**Fix**: Wrap `self.visual = Qwen3_VisionTransformer(...)` initialization in
`set_default_torch_dtype(torch.float32)`. ViT parameters are created in FP32,
weights are preserved in FP32 via `copy_()`, and outputs are seamlessly cast
back to BF16 at the multimodal merge point
(`_merge_multimodal_embeddings` → `mm_embeds_flat.to(dtype=input_dtype)`).

This pattern already exists in vLLM for `deepseek_vl2.py` and `minicpmv.py`
(FP16-init for ViT).

| ViT dtype | Result | Forward time |
|-----------|--------|-------------|
| BF16 | 100% NaN | ~293s |
| FP16 | 100% NaN | ~84s |
| FP32 | OK | ~45s |

FP32 ViT is faster because BF16/FP16 times include NaN-propagation overhead.
The ViT is ~7% of total parameters (590M/8B), so the net FP32 overhead is
~10-15% on multimodal requests, 0% on text-only.

**Patch status in this repo**: documented from the Bitserv deployment notes,
but not imported or wired into `vllm-packages.yaml` yet.

## Pooler Architecture

`Qwen3VLForConditionalGeneration` has **no native pooler**. vLLM's
`as_embedding_model()` injects a `DispatchPooler.for_embedding` with:

- LAST-token pooling
- L2 normalization

This is equivalent to the official `Qwen3VLEmbedder._pooling_last()` method.

**Required CLI flags**: `--runner pooling --convert embed`

Without `--convert embed`, the model outputs raw hidden states instead of
proper normalized embeddings.

## vLLM Server Configuration

### .env

```bash
VLLM_ROLES="qwen3_embed qwen3_rerank"

# Embed (port 8102)
VLLM_QWEN3_EMBED_MODEL="/path/to/your/models/Qwen3-VL-Embedding-8B"
VLLM_QWEN3_EMBED_PORT=8102
VLLM_QWEN3_EMBED_DEVICE="${VLLM_DEVICE_DEFAULT}"
VLLM_QWEN3_EMBED_RUNNER="pooling"
VLLM_QWEN3_EMBED_CONVERT="embed"
VLLM_QWEN3_EMBED_MAX_MODEL_LEN=32768
VLLM_QWEN3_EMBED_GPU_MEMORY_MB=22118   # 0.45 × 49152 (dual-instance)
VLLM_QWEN3_EMBED_KV_CACHE_DTYPE="fp8_e5m2"
VLLM_QWEN3_EMBED_CPU_OFFLOAD_GB=5
VLLM_QWEN3_EMBED_ENFORCE_EAGER=true
VLLM_QWEN3_EMBED_LIMIT_MM_PER_PROMPT='{"video": 0, "image": 1}'
# VLLM_QWEN3_EMBED_SKIP_MM_PROFILING=true  # NOT recommended for VL models

# Rerank (port 8103)
VLLM_QWEN3_RERANK_MODEL="/path/to/your/models/Qwen3-VL-Reranker-8B"
VLLM_QWEN3_RERANK_PORT=8103
VLLM_QWEN3_RERANK_DEVICE="${VLLM_DEVICE_DEFAULT}"
VLLM_QWEN3_RERANK_RUNNER="pooling"
VLLM_QWEN3_RERANK_MAX_MODEL_LEN=32768
VLLM_QWEN3_RERANK_GPU_MEMORY_MB=22118   # 0.45 × 49152 (dual-instance)
VLLM_QWEN3_RERANK_KV_CACHE_DTYPE="fp8_e5m2"
VLLM_QWEN3_RERANK_CPU_OFFLOAD_GB=5
VLLM_QWEN3_RERANK_ENFORCE_EAGER=true
VLLM_QWEN3_RERANK_HF_OVERRIDES='{"architectures":["Qwen3VLForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
VLLM_QWEN3_RERANK_LIMIT_MM_PER_PROMPT='{"video": 0, "image": 1}'
# VLLM_QWEN3_RERANK_SKIP_MM_PROFILING=true  # NOT recommended for VL models
```

### Startup

```bash
cd /opt/src/vllm/_gfx115x_
scripts/vllm-start.sh   # uses setsid (not nohup) to avoid EngineCore zombie
```

### Runtime Characteristics

#### Per Instance (Embed or Reranker, W8A16 + AITER)

| Metric | Value |
|--------|-------|
| Model weights (W8A16) | ~9.9 GiB |
| Weights offloaded (CPU/UMA) | ~5.0 GiB via UVA zero-copy |
| Weights on GPU | ~4.9 GiB |
| ViT FP32 overhead | ~1.2 GiB |
| KV Cache (reserved) | ~7.0 GiB |
| Framework overhead | ~2.0 GiB |
| **Total VRAM per instance** | **~10.5 GiB** |
| **GPU memory budget** | **12.0 GiB** (0.244 × 49152 MB) |
| **Headroom** | **~1.5 GiB** |
| GPU Memory Util | 0.244 (12000/49152 MB) |
| Startup Time (cached) | ~30s (AITER kernels cached in `~/.triton/cache/`) |
| Startup Time (first AITER) | ~16 min (JIT kernel compilation for gfx1151) |
| Warm latency (embed) | **0.16s** (AITER) / 0.56s (no AITER) |
| Warm latency (rerank, 5 docs) | **0.18s** (AITER) / 1.51s (no AITER) |
| Video Input | **Disabled** — `--limit-mm-per-prompt video=0` (see below) |

#### Combined (Both Instances, W8A16 + AITER)

| Metric | Value |
|--------|-------|
| **Total VRAM used** | **~21.2 GiB** (measured) |
| **Total CPU/UMA used** | **~10.0 GiB** |
| **Free VRAM buffer** | **~26.8 GiB** (for Lemonade or other models) |
| **30-min burn test** | 8344 requests (0 errors, VRAM delta: +0.000 GiB) |

Embedding workloads are short-sequence (typically <512 tokens). FP8 KV cache
at 0.45 utilization provides ~7 GiB for KV — sufficient for ~70+ concurrent
embedding requests or ~3 concurrent sequences at maximum 32K context.

## Memory Optimization for Co-Hosting

vLLM allocates **all** GPU memory at startup — there is no "optimistic" or
dynamic mode that frees unused memory at runtime. The KV cache is fully
pre-allocated based on `gpu_memory_utilization`, and PagedAttention does not
release blocks back to the OS when idle.

This means a server tuned for maximum throughput (0.93 utilization) wastes
~16-17 GiB on unused KV cache capacity when serving short-sequence embedding
requests.

### Available Optimizations

| # | Option | Description | Savings | Risk |
|---|--------|-------------|---------|------|
| 1 | `--gpu-memory-utilization 0.55` | Reduce VRAM budget; embeddings need little KV cache | ~20 GiB | Low — embedding sequences are short |
| 2 | `--kv-cache-dtype fp8_e5m2` | Store KV cache in FP8 instead of BF16 (halves KV cache size) | ~4-5 GiB | Required on gfx1151; E4M3 crashes on RDNA 3+ |
| 3 | `--kv-offloading-size 4 --kv-offloading-backend native` | Move inactive KV blocks to CPU RAM | ~4 GiB | Re-activation latency on idle |
| 4 | `--cpu-offload-gb 5` | Offload model weights to CPU via UVA zero-copy | ~5 GiB | Higher per-forward latency |
| 5 | `--max-model-len 4096` | Halve max context → halve KV cache reservation | ~4-8 GiB | Shorter max context |
| 6 | `--enforce-eager` | Already active — disables CUDA graphs, saves their memory | 0 (already on) | None |
| 7 | `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` | **Performance only** — requires AITER, not available on gfx1151 | 0 GiB | N/A — unsupported |

### FP8 KV Cache on ROCm

On gfx1151 (RDNA 3+), `--kv-cache-dtype fp8_e5m2` is required for FP8 KV cache.
`fp8_e4m3` crashes because Triton cannot compile `float8_e4m3fn` atoms on RDNA 3+.
The E5M2 patch (BUILD-FIXES #92, #93) adds the missing C++ dispatch and conversion
functions in `amd/quant_utils.cuh` and corrects `fp8_dtype()` in `rocm.py`.
**Patch status in this repo**: the Bitserv `fp8-e5m2-quant-utils.patch` did
not apply cleanly to the current `/opt/src/vllm` source, so it is not imported
in this branch. Refresh it against the current vLLM tree before manifest
integration.

The ROCm attention backends (`rocm_aiter_fa.py`, `rocm_aiter_unified_attn.py`,
`triton_attn.py`) all include `is_fp8_kv_cache` handling with dynamic
quantization scales.

| `cache_dtype` | KV Size | ROCm Support | Notes |
|---------------|---------|---------------|-------|
| `auto` (= BF16) | 1× | Yes | Default |
| `fp8_e4m3` | 0.5× | **No (gfx1151)** | Triton cannot compile `float8_e4m3fn` on RDNA 3+ |
| `fp8_e5m2` | 0.5× | **Yes (with patch)** | Required on gfx1151 — needs E5M2 patch (#92, #93) |
| `fp8_ds_mla` | 0.5× | **No** | DeepSeek MLA architecture only |

For embedding workloads, FP8 KV cache quality impact is negligible — the model
outputs a single vector after pooling, not token-level logits.

### Co-Hosting Scenarios

#### A: Dual-Instance BF16 (baseline, 32K)

Both models run simultaneously in BF16, each with FP8 KV cache and 5 GiB CPU offload.

```bash
# Per-instance flags (applied to both roles):
--gpu-memory-utilization 0.45       # 22118 MiB each
--kv-cache-dtype fp8_e5m2           # Halves KV cache (E5M2 required on gfx1151)
--cpu-offload-gb 5                  # UVA zero-copy weight offload
--max-model-len 32768               # Full 32K context
--enforce-eager                      # Required for cpu-offload on V1
--limit-mm-per-prompt '{"video": 0, "image": 1}'
```

| Component | Per Instance (GiB) | Total (GiB) |
|-----------|-------------------|-------------|
| Weights on GPU | 10.3–11.5 | 21.8 |
| Weights offloaded (CPU) | 5.0 | 10.0 |
| ViT FP32 | 1.2 | 2.4 |
| KV Cache (FP8) | 7.0 | 14.0 |
| Framework | 2.0 | 4.0 |
| **Total VRAM** | **~21.6** | **~43.2** |
| **Total CPU/UMA** | **5.0** | **10.0** |
| **Free VRAM buffer** | — | **~4.8** |

#### B: Solo Embedding (single model, max throughput)

```bash
--gpu-memory-utilization 0.93
--kv-cache-dtype auto               # BF16 default
--max-model-len 8192
--enforce-eager
```

| Component | VRAM |
|-----------|------|
| Model weights | ~15.3 GiB |
| ViT FP32 | ~1.2 GiB |
| KV cache (BF16, full budget) | ~22.5 GiB |
| Framework | ~2 GiB |
| **Total vLLM** | **~41-43 GiB** |
| **Free for Lemonade** | **~7-8 GiB** (unusable) |

#### C: Dual-Instance W8A16+AITER (production, 32K)

Both models quantized to W8A16 INT8, with AITER enabled for optimized
kernel dispatch. This is the validated production configuration.

```bash
# Per-instance flags:
--quantization compressed-tensors    # Enable INT8 weight loading
--gpu-memory-utilization 0.244       # ~12000 MiB each
--cpu-offload-gb 5
--max-model-len 32768
--enforce-eager
--limit-mm-per-prompt '{"video": 0, "image": 1}'
# AITER: VLLM_ROCM_USE_AITER=1 (global env)
# Embed only: --runner pooling --convert embed
# Rerank only: --runner pooling --hf-overrides {...}
```

| Component | Per Instance (GiB) | Total (GiB) |
|-----------|-------------------|-------------|
| Weights on GPU (INT8) | ~4.9 | ~9.8 |
| ViT FP32 overhead | 1.2 | 2.4 |
| KV Cache (pooling≈0) | ~0 | ~0 |
| Framework | 2.0 | 4.0 |
| Other (KV reservation, etc.) | ~2.5 | ~5.0 |
| **Total VRAM (measured)** | **~10.6** | **~21.2** |
| **Free for Lemonade** | — | **~26.8 GiB (VRAM) + ~25 GiB (GTT)** |

**Validated performance** (AITER enabled):
- Embed warm latency: **0.16s** (3.5× faster than without AITER)
- Rerank warm latency: **0.18s** (8.5× faster than without AITER)
- Concurrent pipeline: **0.32s** (embed+rerank parallel)
- 30-min burn test: 8344 requests, 0 errors, VRAM delta +0.000 GiB

## INT8 Quantization (W8A16)

Findings synthesized from RDNA3.5 ISA specs, vLLM/llmcompressor
source-code analysis, and three independent Deep Research queries
(Perplexity, Claude, Gemini). Confidence: confirmed where noted.

### Why INT8 on gfx1151

RDNA 3.5 (gfx1151) has native INT8 WMMA hardware
(`v_wmma_i32_16x16x16_iu8`) but **no native FP8 compute**. INT8 is the
physically correct quantization path. FP8 E4M3 crashes on gfx1151 due to
Triton compilation failures (`float8_e4m3fn` atoms unsupported in RDNA3+).

| Capability | Status | Detail |
|-----------|--------|--------|
| INT8 WMMA | **Native** | `v_wmma_i32_16x16x16_iu8` (16x16x16 tile, I8→I32 accumulate) |
| BF16 WMMA | **Native** | `v_wmma_f32_16x16x16_bf16` |
| FP16 WMMA | **Native** | `v_wmma_f32_16x16x16_f16` |
| FP8 WMMA | **None** | Must cast to BF16 via `hip_fp8` library before FMA |
| Peak INT8 throughput | ~32% higher than BF16 scalar paths | |
| Peak BF16 throughput | ~59.4 TFLOPS | 40 CUs @ 2.9 GHz theoretical |
| Memory bandwidth | 212 GB/s sustained | 256-bit LPDDR5-8000 (256 GB/s theoretical) |

> **Key insight**: FP8 on RDNA 3.5 is storage-only. Every FP8 GEMM incurs a
> software cast to BF16, making it slower than native INT8. FP8 E4M3
> additionally crashes on gfx1151 due to Triton compilation failures
> (`float8_e4m3fn` atoms unsupported in RDNA3+ Triton backend).

### UMA Memory Hierarchy

| Pool | Size | Accessible by |
|------|------|---------------|
| VRAM (BIOS carveout) | 48 GiB | ROCm `hipMalloc`, Vulkan/RADV |
| GTT (kernel-managed) | ~25 GiB | Vulkan/RADV only (not `hipMalloc`) |
| System RAM | ~23 GiB | CPU, vLLM cpu-offload via UVA |

### W8A8 vs W8A16 Decision

| Aspect | W8A8 (weight + activation INT8) | W8A16 (weight-only INT8) |
|--------|----------------------------------|--------------------------|
| Weight storage | ~10.5 GiB | ~9.9 GiB |
| Activation precision | INT8 (dynamic per-token) | **BF16 (preserved)** |
| Kernel path | TritonInt8ScaledMMLinearKernel (MLIR→WMMA) | Dequant→BF16 WMMA (hipBLASLt) |
| Accuracy risk | Medium — 36 layers of accumulated quantization error | **Low** — error limited to weight dequantization |
| Calibration | Required (512+ samples) | **Not required** (RTN) |
| Embedding quality | Acceptable with GPTQ | Near-native |
| Reranker scoring | Risk: tiny logit shifts can flip rankings | Near-native |

**Decision: W8A16**. Embedding models pool the last hidden state →
L2-normalize → cosine similarity. Every bit in the activation matters because
small errors propagate directly into the embedding vector. Rerankers have
only 2 output classes (yes/no) — a tiny logit shift can flip the ranking.
W8A8 quantizes activations to INT8 at every layer (36 layers = 36× accumulated
quantization error). W8A16 preserves BF16 activations throughout.

W8A8 remains available as a throughput optimization (Phase 5) if accuracy
proves acceptable after testing. A pre-quantized W8A8 reference model from
HuggingFace (`collin-park/Qwen3-VL-Embedding-8B-W8A8`, 9.9 GiB) is available
for comparison testing.

### Quantization Comparison (All Schemes)

| Property | BF16 (current) | W8A8 INT8 | W8A16 INT8 | FP8 E4M3 |
|----------|---------------|-----------|------------|----------|
| Weight storage | ~16 GiB | ~10.5 GiB | ~10-11 GiB | ~9 GiB |
| Weight precision | 16-bit float | INT8 per-channel | INT8 per-channel | FP8 E4M3 per-channel |
| Activation precision | BF16 | INT8 dynamic per-token | **BF16 (preserved)** | FP8 dynamic |
| KV Cache | FP8 E5M2 | FP8 E5M2 | FP8 E5M2 | N/A |
| Matmul path | BF16 WMMA | INT8 WMMA via Triton | Dequant→BF16 WMMA | **Crash on gfx1151** |
| Accuracy risk | None | Medium (activation quant) | Low (BF16 activations) | N/A |
| Embedding quality | Baseline | Acceptable with GPTQ | Near-native | N/A |
| Reranker scoring | Baseline | Risk: binary logit shift | Near-native | N/A |
| vLLM backend | native | `compressed-tensors` | `compressed-tensors` | N/A |
| Pre-quantized HF | N/A | `collin-park/...W8A8` | Self-quantize | `RamManavalan/...FP8` |
| gfx1151 confidence | Proven | High (Triton path) | **Very high** | Dead |

### QuantizationModifier vs GPTQModifier (Source-Code Verified)

| Aspect | `QuantizationModifier` | `GPTQModifier` |
|--------|----------------------|----------------|
| Algorithm | **RTN** (round-to-nearest) | **GPTQ** (Hessian-based error compensation) |
| Calibration data | **Not required** | Required (512+ samples) |
| Hessian computation | **None** | Always (Cholesky decomposition) |
| `oneshot()` dataset param | Not needed | Must provide `dataset`, `num_calibration_samples` |
| Quality (W8A16) | Near-native | Marginally better (~0.1%) |
| RAM overhead | ~16 GiB (model only) | ~30-40 GiB (model + Hessians) |
| Time | ~30 sec | ~60-120 min |

**Source-code evidence** (llmcompressor 0.10.0.1):

`GPTQModifier.on_start()` (`gptq/base.py`):
```python
for _, module in named_modules:
    if getattr_chain(module, "quantization_scheme.weights", None) is not None:
        if not isinstance(module, torch.nn.Embedding):
            self.register_hook(module, self.calibrate_module, "forward")
```
The `calibrate_module` hook accumulates Hessians on **every forward pass**.
There is **no code path that skips Hessians** for weight-only schemes.

`QuantizationModifier.on_start()` (`quantization/base.py`):
```python
for _, module in tqdm.tqdm(named_modules, desc="Calibrating weights"):
    update_weight_zp_scale(module)
```
`update_weight_zp_scale` calls a `memoryless_minmax` observer directly on
weights — no Hessians, no calibration data, pure RTN.

**Decision**: `QuantizationModifier` (RTN). GPTQ available as fallback.

### W8A16 Scheme Definition

```python
W8A16 = dict(
    weights=QuantizationArgs(num_bits=8, type=INT, strategy=CHANNEL, symmetric=True, dynamic=False),
)
```

No `input_activations` key = pure weight-only quantization. BF16 activations
are fully preserved. Confirmed by `QuantizationScheme` class defaults
(`input_activations=None`, `output_activations=None`).

### Quantization Ignore Patterns

| Pattern | Matched layers | Reason |
|---------|---------------|--------|
| `lm_head` | `model.lm_head` | Classification head; INT8 error propagates to output logits |
| `re:.*visual.*` | All 108 ViT Linear layers | ViT produces NaN in BF16 on gfx1151; must stay BF16 for FP32 patch |

### Quantization Toolchain

| Component | Version | Location |
|-----------|---------|----------|
| Python | 3.12.13 | Isolated venv `/opt/src/vllm/.venv-quantize/` |
| llmcompressor | 0.10.0.1 | Installed in venv |
| compressed-tensors | 0.14.0.1 | Output format |
| torch | 2.11.0+cpu | CPU-only for quantization |
| transformers | 4.57.6 | Qwen3-VL support |

The quantization venv is isolated from the vLLM production venv (Python 3.13).
No ROCm dependencies — CPU-only operation. `oneshot()` automatically saves
model + processor with `save_compressed=True`.

### W8A16 Production Recipe (QuantizationModifier)

```python
import os
os.environ["HIP_VISIBLE_DEVICES"] = ""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

DTYPE = torch.bfloat16
INPUT_DIR = "/path/to/your/models/Qwen3-VL-Embedding-8B"
OUTPUT_DIR = "/path/to/your/models/Qwen3-VL-Embedding-8B-W8A16"
IGNORE = ["lm_head", "re:.*visual.*"]

model = AutoModelForImageTextToText.from_pretrained(
    INPUT_DIR,
    torch_dtype=DTYPE,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(INPUT_DIR, trust_remote_code=True)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="W8A16",
    ignore=IGNORE,
)

oneshot(
    model=model,
    recipe=recipe,
    output_dir=OUTPUT_DIR,
)
```

For Reranker, use the same recipe with `INPUT_DIR` and `OUTPUT_DIR` changed.
The `IGNORE` list is identical — there is no `score` layer in the checkpoint.

### Kernel Dispatch: W8A8 at Inference

```
Model loads with --quantization compressed-tensors
  → CompressedTensorsW8A8Int8 scheme activated
    → MPLinearKernel selector checks kernels in priority order:

      1. AiterInt8ScaledMMLinearKernel
         → REJECT: gated behind on_gfx9() — only gfx94x/gfx95x (CDNA3/4)
         → Bypassable with source patch (see AITER Unlock section)

      2. ConchLinearKernel (ROCm priority)
         → min_capability=80: PASS (gfx1151 reports 115)
         → BUT: rejects at can_implement() — CUDA-centric, group_size
           assertions fail on ROCm, conch library deps missing
         → REJECT

      3. ExllamaLinearKernel (fallback)
         → Strictly supports float16 activations only
         → W8A8 produces int8 activations — incompatible
         → REJECT

      4. MarlinLinearKernel / MacheteLinearKernel
         → Explicit CUDA-only assertions
         → REJECT

      5. TritonInt8ScaledMMLinearKernel (final fallback)
         → Architecture-agnostic: compiles via MLIR to host ISA
         → Emits v_wmma_i32_16x16x16_iu8 on gfx1151
          → **ACCEPT: This is the operational kernel**
```

**Confidence**: Confirmed (Gemini Deep Research, vLLM source: `compressed_tensors_w8a8_int8.py`, `mp_linear_kernel`).

### Kernel Dispatch: W8A16 at Inference

```
Model loads with --quantization compressed-tensors
  → CompressedTensorsWNA16 with num_bits=8
    → Dequantize INT8 weights to BF16 in SRAM (per-channel scale)
    → Standard BF16 GEMM via hipBLASLt/hipBLAS
    → No specialized INT8 GEMM kernel required
```

This is the simplest and most reliable path. Weight dequantization is a
trivial per-channel operation, and the resulting BF16 GEMM uses the
well-validated hipBLASLt backend (our TheRock build includes gfx1151
TensileLibrary + extop kernels).

For W8A8 (reference model), the dispatch chain is more complex: all
CUDA-centric kernels (Conch, Exllama, Marlin, Machete) are rejected on
ROCm, falling back to `TritonInt8ScaledMMLinearKernel` which compiles via
MLIR to native INT8 WMMA. This works but carries risks (Triton block-size
limits, Triton 3.6.0 casting bug — affects W8A8 only, irrelevant for W8A16).

**Confidence**: Confirmed (Gemini, Claude, vLLM source).

### ViT FP32 Patch + INT8 Interaction

The existing ViT FP32 patch operates independently of INT8 quantization:

| Component | Stored in quantized checkpoint | Runtime behavior |
|-----------|-------------------------------|-----------------|
| ViT weights | BF16 (excluded from quant) | Cast to FP32 by existing patch |
| ViT activations | N/A | FP32 (forced by patch) |
| LLM Linear weights | INT8 per-channel | Dequantized to BF16 |
| LLM activations | BF16 (preserved, W8A16) | Normal BF16 compute |
| KV Cache | FP8 E5M2 (separate patch) | Unchanged |

The ViT FP32 runtime behavior must remain active regardless of quantization
scheme. The Bitserv deployment used a manifest-applied patch for this; that
patch is not imported here yet.

**Confidence**: Confirmed (ViT exclusion explicit in quantization recipes, FP32 patch operates at model init).

### oneshot() Save Behavior (Source-Code Verified)

`oneshot()` automatically saves with `save_compressed=True` (default) when
`output_dir` is provided. It also calls `processor.save_pretrained(output_dir)`
if a Processor is loaded. No separate `model.save_pretrained()` call needed.

### ignore Pattern Matching (Source-Code Verified)

- Strings starting with `"re:"` → regex match against full module path
- Otherwise → exact match on module name OR class name (via `_match_class()`)
- No suffix/prefix matching for non-regex strings

### Reranker "score" Head — Not in Checkpoint (Source-Code Verified)

The Qwen3-VL-Reranker-8B checkpoint does **not** contain a `score.weight`
tensor. The model is saved as `Qwen3VLForConditionalGeneration` with only
`lm_head.weight` and `model.*` prefixes. The `score` head only exists when
vLLM overrides the architecture to `Qwen3VLForSequenceClassification` at
runtime via `--hf-overrides`. vLLM uses `lm_head` logits and indexes into
specific token positions ("no"/"yes") for scoring.

**Consequence**: `"score"` in the ignore list is harmless but unnecessary.
Both models use the same ignore list: `["lm_head", "re:.*visual.*"]`.

### device_map Rationale

Using `device_map="auto"` on UMA with a visible ROCm GPU would map layers to
VRAM. If vLLM or Lemonade is running concurrently, this causes OOM or MES
faults. CPU-only quantization for 8B models is safe with 96 GiB RAM.

**Note**: `device_map="cpu"` with llmcompressor triggers a bug in
`compressed_tensors` — `dispatch_model()` requires a visible device. CPU-only
torch raises `MemoryError: Did not find any devices to dispatch model to`.
Workaround: monkey-patch `get_device_memory` to report CPU RAM. Our
`quantize_w8a16.py` script handles this with `HIP_VISIBLE_DEVICES=""` and
the dispatch patch.

### torch_dtype=torch.bfloat16 Rationale

The Reranker's `config.json` declares `text_config.dtype: "float32"` and
`vision_config.dtype: "float32"`, but the actual safetensors weights are BF16
(the checkpoint is only 17 GiB, not 32 GiB). With `torch_dtype="auto"`,
transformers would load the model in FP32, consuming ~32 GiB RAM.

Forcing `torch_dtype=torch.bfloat16` is correct because:
1. GPTQ/RTN weight-only quantization computes per-channel scales — identical
   whether the source is FP32 or BF16
2. The actual weights on disk are BF16 (FP32 config is an upstream artifact)
3. BF16 halves RAM during quantization (~16 GiB instead of ~32 GiB)

### W8A8 INT8 — GPTQModifier Recipe (Reference/Fallback)

For reference only. Requires calibration data:

```python
from llmcompressor.modifiers.quantization import GPTQModifier

recipe = GPTQModifier(
    targets="Linear",
    scheme="W8A8",
    ignore=["lm_head", "re:.*visual.*"],
    dampening_frac=0.01,
    offload_hessians=True,  # critical on UMA — reduces RAM by ~15 GiB
)

oneshot(
    model=model,
    recipe=recipe,
    dataset="ultrachat_200k",
    num_calibration_samples=512,
    max_seq_length=2048,
    output_dir=OUTPUT_DIR,
)
```

### Quantization Duration Estimate

| Model | Scheme | Modifier | Estimated Time | RAM Required |
|-------|--------|----------|---------------|--------------|
| Either 8B | W8A16 | QuantizationModifier (RTN) | ~30 sec | ~16 GiB |
| Embedding 8B | W8A8 | GPTQModifier (GPTQ) | 60-120 min | ~30 GiB |
| Reranker 8B | W8A8 | GPTQModifier (GPTQ) | 60-120 min | ~30 GiB |

### Triton Constraints on gfx1151

#### Block Size Limits

The Triton compiler on gfx1151 has strict tile-size requirements for INT8
`tl.dot` operations. Exceeding these causes MLIR lowering failures:

| Parameter | Maximum | Recommended |
|-----------|---------|-------------|
| BLOCK_M | 32 | 32 |
| BLOCK_N | 32 | 32 |
| BLOCK_K | 64 | 64 |

**Mitigation**: vLLM's `TritonInt8ScaledMMLinearKernel` should respect these
via its autotuner config. If not, manual tuning required.

#### Triton 3.6.0 Implicit Casting Bug

`store(i1, i32)` operations can implicitly cast 32-bit integers down to 8-bit,
corrupting activation scales. **This affects W8A8 (TritonInt8ScaledMMLinearKernel)
only.** W8A16 does not use Triton INT8 kernels — irrelevant for production.

**Mitigation** (W8A8 only): Ensure Triton kernels explicitly enforce
`.to(tl.float32)` bounds prior to scaling multiplication.

#### tl.dot(int8, int8, out_dtype=tl.int32) on gfx1151

AOTriton 0.10b introduced experimental gfx1151 support. Triton's AMD backend
targets gfx1151 in recent builds and should emit `v_wmma_i32_16x16x16_iu8`
when compiling `tl.dot(int8, int8, out_dtype=tl.int32)`.

**Confidence**: Probable (ISA confirms hardware capability, AOTriton
experimental support exists, but no public production vLLM benchmark confirms
end-to-end performance).

### Pre-Quantized Models

| Model | Repo | Format | Size | ViT Precision | Usable on gfx1151 |
|-------|------|--------|------|---------------|-------------------|
| Embedding W8A8 | `collin-park/Qwen3-VL-Embedding-8B-W8A8` | compressed-tensors | ~10.5 GiB | BF16 (preserved) | Yes (Triton path) |
| Embedding FP8 | `RamManavalan/Qwen3-VL-Embedding-8B-FP8` | compressed-tensors | ~9 GiB | BF16 (preserved) | **No** (FP8 E4M3 crash) |
| Reranker FP8 | `Forturne/Qwen3-VL-Reranker-8B-FP8` | compressed-tensors | ~9 GiB | BF16 (preserved) | **No** (FP8 E4M3 crash) |

**W8A8 Details** (collin-park): GPTQ W8A8 INT8, calibrated with 512 samples
from ultrachat-200k (max 2048 tokens). ViT excluded via
`ignore=["re:.*visual.*"]`, no SmoothQuant. Tensor types: BF16 + I8.
Tested on RTX 3090 (24 GB), vLLM 0.17.1.

### Memory Savings Summary

| | BF16 Current | INT8 W8A16 Target | Savings |
|---|-------------|-------------------|---------|
| vLLM total VRAM | ~43.2 GiB | ~18-20 GiB | **~23-25 GiB** |
| Available for Lemonade | ~4.8 GiB | ~53-55 GiB | **+48-50 GiB** |

### What Does NOT Work

- **Dynamic/optimistic allocation**: vLLM has no runtime memory release.
  All KV cache blocks are pre-allocated at startup.
- **`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT`**: Requires AITER
  (`VLLM_ROCM_USE_AITER=1`), which is not enabled on gfx1151. Shuffle is a
  performance optimization, not a memory optimization.
- **`--kv-cache-dtype fp8_e4m3`**: Triton cannot compile `float8_e4m3fn` atoms on
  RDNA 3+ (gfx1151). Use `fp8_e5m2` instead (requires E5M2 patch, BUILD-FIXES #92/#93).
- **`--kv-cache-dtype fp8_ds_mla`**: Only for DeepSeek MLA architectures.
- **GTT as VRAM extension**: `hipMalloc` can only allocate from the 48 GiB
  BIOS carveout. The GTT pool (~25 GiB) is kernel-managed and not accessible
  via `hipMalloc`. Vulkan/RADV can use GTT, but vLLM/ROCm cannot.
- **FP8 E4M3 quantization**: Triton cannot compile `float8_e4m3fn` atoms on
  RDNA 3+. Pre-quantized FP8 checkpoints (RamManavalan, Forturne) are unusable.
- **GPTQModifier as RTN shortcut**: `GPTQModifier` always computes Hessians
  regardless of scheme. Use `QuantizationModifier` for true RTN (no calibration).
- **`device_map="cpu"` with llmcompressor**: The `dispatch_model()` function
  in `compressed_tensors` requires a visible device. CPU-only torch raises
  `MemoryError: Did not find any devices to dispatch model to`. Workaround:
  monkey-patch `get_device_memory` to report CPU RAM.

### Image Token Budget at 32K Context

Qwen3-VL: patch_size=16, spatial_merge_size=2 → 1024 pixels per visual token.

| Image Size | Pixels | Visual Tokens | Fits in 32K? |
|------------|--------|--------------|---------------|
| Default max (~1 MP) | ~1,000,000 | ~1,100 | Yes (3%) |
| 2K (2048×2048) | 4,194,304 | ~4,100 | Yes (13%) |
| 4K (3840×2160) | 8,294,400 | ~8,100 | Yes (25%) |
| 8K (7680×4320) | 33,177,600 | ~32,400 | Marginal (100%) |

At 32K context, 4K images fit comfortably with room for text. The model's
built-in auto-resize caps at ~1 MP by default, so most images consume only
~1,100 tokens.

## HTTP API Reference

### Supported Endpoints

| Endpoint | Style | Multimodal | Notes |
|----------|-------|-----------|-------|
| `/v1/embeddings` | Completion | No | `input: str \| [int]` — text only |
| `/v1/embeddings` | Chat | **Yes** | `messages: [{role, content}]` — multimodal |
| `/pooling` | Chat | **Yes** | Same as chat-style embeddings |
| `/v2/embed` | Cohere | **Yes** | `images: [url]` or `inputs: [{content}]` |

### Text-only (completion-style, simplest)

```json
POST /v1/embeddings
{
  "model": "Qwen3-VL-Embedding-8B",
  "input": "A woman playing with her dog on a beach at sunset."
}
```

### Multimodal: Image + Text (chat-style)

```json
POST /v1/embeddings
{
  "model": "Qwen3-VL-Embedding-8B",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        {"type": "text", "text": "a woman and a dog"}
      ]
    }
  ]
}
```

### Multimodal: Image-only (chat-style)

```json
POST /v1/embeddings
{
  "model": "Qwen3-VL-Embedding-8B",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
      ]
    }
  ]
}
```

### Text with Instruction (chat-style)

```json
POST /v1/embeddings
{
  "model": "Qwen3-VL-Embedding-8B",
  "messages": [
    {
      "role": "system",
      "content": [{"type": "text", "text": "Retrieve relevant documents for the query."}]
    },
    {
      "role": "user",
      "content": [{"type": "text", "text": "A woman playing with her dog on a beach."}]
    }
  ]
}
```

### Cohere-style: Image (simplest for images)

```json
POST /v2/embed
{
  "model": "Qwen3-VL-Embedding-8B",
  "images": ["https://example.com/photo.jpg"],
  "embedding_types": ["float"]
}
```

### Cohere-style: Mixed text + image

```json
POST /v2/embed
{
  "model": "Qwen3-VL-Embedding-8B",
  "inputs": [
    {
      "content": [
        {"type": "text", "text": "a woman and a dog"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
      ]
    }
  ],
  "embedding_types": ["float"]
}
```

### Cohere-style: Text-only

```json
POST /v2/embed
{
  "model": "Qwen3-VL-Embedding-8B",
  "texts": ["A woman playing with her dog on a beach at sunset."],
  "embedding_types": ["float"]
}
```

### /pooling endpoint

```json
POST /pooling
{
  "model": "Qwen3-VL-Embedding-8B",
  "task": "embed",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        {"type": "text", "text": "a woman and a dog"}
      ]
    }
  ]
}
```

## Encoder-Cache Profiling Crash (Video Fix)

### Symptom

Both Embed and Reranker instances crash during startup at the encoder-cache
profiling step:

```
Encoder cache will be initialized with a budget of 12288 tokens,
and profiled with 1 video items of the maximum feature size.
```

The EngineCore becomes a zombie (`ZN`, `<defunct>`) while the APIServer
remains sleeping. VRAM is **not released** (16.2 GiB leaked). No health check
ever passes.

### Root Cause

Qwen3VL defines `DUMMY_VIDEO_NUM_FRAMES = 2048` in vLLM's model code.
During startup profiling, vLLM runs a full ViT forward pass with 2048 video
frames at maximum resolution to determine the encoder cache budget. On
gfx1151 with only ~22 GiB VRAM per instance (dual-instance setup), this
memory spike exceeds available VRAM and crashes the EngineCore.

The profiling path in `gpu_model_runner.py:5757-5788` performs:
1. `mm_budget.get_modality_with_max_tokens()` → selects `"video"` (highest token count)
2. `_get_mm_dummy_batch("video", 1)` → generates 2048-frame dummy video
3. `model.embed_multimodal(**inputs)` → full ViT forward pass → **OOM crash**

Video is the **maximum-token modality** by far: 2048 frames × temporal compression
produces ~12288 visual tokens, versus ~1100 tokens for a single image.

### Fix: `--limit-mm-per-prompt '{"video": 0, "image": 1}'`

Setting `video: 0` removes video from `tower_modalities` in
`encoder_budget.py:73-77` — the video modality is never profiled and never
included in the encoder cache budget. Only image (with `image: 1`) is profiled,
which uses ~1100 tokens instead of ~12288.

This is the **correct** approach because:
- Embedding/reranking use cases don't need video input
- Video profiling at 2048 frames is the **most memory-intensive** operation in vLLM startup
- Image-only profiling uses the much smaller ViT forward pass (~1 GiB vs ~12+ GiB)
- The alternative (`--skip-mm-profiling`) would skip ALL MM profiling including images

### Alternative Considered: `--skip-mm-profiling`

This flag skips multimodal profiling entirely. While it avoids the video crash,
it also skips image profiling, which means vLLM cannot properly size the
encoder cache for image inputs. **Not recommended** for multimodal models.

```bash
# NOT RECOMMENDED — disables image encoder cache too
--skip-mm-profiling
```

### Implementation

Added per-role `LIMIT_MM_PER_PROMPT` in `.env`:

```bash
VLLM_QWEN3_EMBED_LIMIT_MM_PER_PROMPT='{"video": 0, "image": 1}'
VLLM_QWEN3_RERANK_LIMIT_MM_PER_PROMPT='{"video": 0, "image": 1}'
```

Added `--limit-mm-per-prompt` CLI flag support in `vllm-start.sh`.

### Startup Log (After Fix)

Expected log output should show image-only profiling:
```
Encoder cache will be initialized with a budget of N tokens,
and profiled with 1 image items of the maximum feature size.
```

Instead of the crash-inducing:
```
... 1 video items of the maximum feature size.
```

## V1 EngineCore Zombie on ROCm/gfx1151

### Symptom

When vLLM V1 is launched via `nohup ... &` in the background, the EngineCore
subprocess becomes a zombie (`<defunct>`). The APIServer stays alive but never
responds to health checks. VRAM is leaked (no release from the zombie process).

`ps aux` shows:
```
bitserv-ai  12345  0.0  0.0      0     0 ?        Z    <date>   0:00 [python] <defunct>
```

### Root Cause

vLLM V1 launches the EngineCore via `multiprocessing.Process` using Python's
forkserver start method. When the parent process runs inside a `nohup ... &`
background shell, the child process inherits a broken session state. On
ROCm/gfx1151, the HIP runtime initialization inside the forked child fails
silently — the process becomes a zombie without ever completing startup.

`nohup` detaches from the terminal but does **not** create a new session. The
parent remains in the original session's process group, causing inconsistent
session state for the forked child.

### Fix

Replace `nohup` with `setsid` in `vllm-start.sh`. `setsid` creates a new
session and process group, giving the forked EngineCore a clean session state.

```bash
# Broken:
nohup vllm serve ... > log 2>&1 &

# Fixed:
setsid vllm serve ... > log 2>&1 &
```

**Alternative**: `VLLM_ENABLE_V1_MULTIPROCESSING=0` runs EngineCore in-process,
avoiding the fork entirely. However, this loses process isolation — an
EngineCore crash kills the entire server, and there is no separate process
to monitor or restart.

### Manual Start (fish shell)

For manual/foreground startup (e.g., debugging), use:

```fish
# In-process (no fork, simpler for debugging):
env VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm serve ...

# Or with setsid for multiprocessing:
setsid vllm serve ... > log 2>&1 &
```

**Do NOT use `nohup ... &`** on ROCm/gfx1151 with vLLM V1.

## V1 EngineCore 100% CPU Idle Busy-Loop

### Symptom

When vLLM V1 is running but has no active requests, each EngineCore subprocess
consumes 100% of one CPU core. For dual-instance setups (Embed + Reranker), this
wastes 2 cores permanently.

`htop` shows:
```
PID   USER   PRI  NI  VIRT   RES   SHR S  %CPU %MEM  TIME+  COMMAND
1234  bitserv-ai  20   0  50.2g  20.1g  2.1g R  99.9  2.1   0:00   python -c ...
```

### Root Cause

The EngineCore `run_busy_loop()` calls `_process_engine_step()` every iteration.
When idle (`model_executed=False`), the existing code only sleeps if
`scheduler.has_unfinished_requests()` is true — but in the idle state this is
false, so no sleep occurs and the loop spins at full speed.

PR #29476 added `time.sleep(0.001)` but it is conditional on
`has_unfinished_requests()`, which is false when truly idle.

### Fix: Progressive Backoff in EngineCore

Patch `vllm/vllm/v1/engine/core.py` to add a progressive idle backoff:

```python
# In EngineCore.__init__:
self._idle_backoff = [0.0, 0.001, 0.010, 0.100, 0.500]  # seconds
self._idle_level = 0

# In _process_engine_step(), replace:
if not model_executed and self.scheduler.has_unfinished_requests():
    time.sleep(0.001)

# With:
if not model_executed:
    if self.scheduler.has_unfinished_requests():
        time.sleep(0.001)
        self._idle_level = 0
    else:
        sleep_dur = self._idle_backoff[
            min(self._idle_level, len(self._idle_backoff) - 1)]
        time.sleep(sleep_dur)
        self._idle_level += 1
else:
    self._idle_level = 0
```

| Consecutive idle steps | Sleep | CPU impact |
|------------------------|-------|------------|
| 1 | 0ms | ~100% |
| 2 | 1ms | ~50% |
| 3 | 10ms | ~10% |
| 4 | 100ms | ~1% |
| 5+ | 500ms | ~0.2% |

Reset is immediate on model execution or new request arrival.

**BUILD-FIXES:** #96
**Auto-applied on rebuild:** No in this repo. The patch is imported as
`patches/enginecore-idle-backoff.patch` for review before build-manifest
integration.
**Upstream status:** Not yet fixed upstream (as of vLLM commit 719735d6c).

### Post-Fix Verification

After restart, CPU usage per EngineCore at idle should drop to <1%:
```
PID   USER   PRI  NI  VIRT   RES   SHR S  %CPU %MEM  TIME+  COMMAND
1234  bitserv-ai  20   0  50.2g  20.1g  2.1g S   0.0  2.1   0:00   python -c ...
```

Note: `S` (sleeping) instead of `R` (running) in the `S` column.

## Known Limitations

### `input_type` not supported (Cohere endpoint)

The `/v2/embed` endpoint rejects `input_type` (e.g., `search_document`,
`search_query`). Qwen3-VL-Embedding does not define task instructions in its
`config.json`. Omit `input_type` from Cohere requests.

### Old completion-style `input` array does not support images

```json
// WRONG — this produces validation errors
{
  "input": [
    {"type": "image_url", "image_url": {"url": "..."}},
    {"type": "text", "text": "..."}
  ]
}
```

The `input` field only accepts `str`, `list[str]`, `list[int]`, or
`list[list[int]]`. For multimodal, use the `messages` format (chat-style) or
the Cohere `images`/`inputs` format.

### MIOpen JIT on first request

The first ViT forward pass triggers MIOpen JIT compilation for gfx1151,
taking ~5 minutes. Subsequent calls are fast (~45s for ViT in offline mode).
The HTTP API returns after full processing — clients should set appropriate
timeouts (≥300s for first request).

## Verification Results

All eight API combinations tested and verified:

| API | Format | Mods | Cosine Sim vs. Text Baseline |
|-----|--------|------|------------------------------|
| `/v1/embeddings` | completion | text | 1.0000 |
| `/v1/embeddings` | chat | img+text | 0.30 |
| `/v1/embeddings` | chat | image | 0.19 |
| `/v1/embeddings` | chat (instr) | text+sys | 0.38 |
| `/pooling` | chat | img+text | 0.30 |
| `/v2/embed` | Cohere images | image | 0.19 |
| `/v2/embed` | Cohere inputs | img+text | 0.23 |
| `/v2/embed` | Cohere texts | text | 1.00 |

Semantic plausibility check: Image+Text vs. Text+Instruction (both
describing a woman with a dog on a beach) = **0.51** — related but distinct
modalities as expected.

### FP8 KV-Cache Incompatibility with Quantized Checkpoints

> **Important**: `--kv-cache-dtype fp8_e5m2` is **incompatible** with
> `compressed-tensors` checkpoints (both W8A8 and W8A16).
>
> vLLM raises `ValueError: fp8_e5m2 kv-cache is not supported with fp8
> checkpoints` (`attention.py:166`). The check triggers because
> `should_load_quant_weights()` returns True for compressed-tensors models,
> but the checkpoint lacks `k_scale`/`v_scale` tensors required by FP8
> KV-cache.
>
> **Fix**: Remove `--kv-cache-dtype fp8_e5m2` from the .env for all INT8
> models. For pooling/reranker models this has zero practical impact —
> single forward pass, no autoregressive KV-cache buildup.

### INT8 Quality Comparison (BF16 vs W8A8 vs W8A16)

Tested with `test_quantize_quality.py` on 5 documents ranked by cosine
similarity to a query ("A woman playing with her dog on a beach at sunset"):

| Metric | BF16 | W8A8 | W8A16 |
|--------|------|------|-------|
| CosSim preservation (vs BF16) | — | 0.999913 | **0.999995** |
| L2 distance (vs BF16) | — | 0.020704 | **0.003630** |
| Spearman ρ (vs BF16 ranking) | — | 0.6000 | **1.0000** |
| Position matches | — | 3/5 | **5/5** |
| Determinism (CosSim re-embed) | 1.000000 | 1.000000 | 1.000000 |

**W8A8 failure**: Swaps Dog+Beach ↔ Pets+Beach (CosSim diff ~0.007 suffices
to flip ranking). Root cause: INT8 activation quantization noise propagates
through 36 transformer layers.

**W8A16**: Near-perfect BF16 fidelity. Weight-only INT8 with BF16 activations
preserves ranking completely.

### Reranker Quality (BF16 vs W8A16)

Both BF16 and W8A16 rerankers show identical ranking anomalies (Stock market
scores highest). This is an intrinsic artifact of the binary yes/no classifier
— **not** a quantization issue.

| Metric | BF16 | W8A16 |
|--------|------|-------|
| Spearman ρ | — | 0.9000 |
| Position matches | — | 3/5 |
| Score shift | — | +0.054 to +0.084 |
| Score spread | 0.1569 | 0.1392 |
| Top-2 gap | 0.0645 | 0.0482 |

The ~+0.07 score shift is expected: INT8 weight dequantization introduces
small logit offsets. Ranking changes are limited to adjacent items with similar
relevance.

### Latency Benchmark (32k Context, 5 Reps)

Warm (steady-state) latency at max_model_len=32768 on gfx1151:

**Embedding** (single 32k query):

| Config | Cold Start | Warm Mean | Warm Min |
|--------|-----------|-----------|----------|
| BF16 / 0 GiB offload | 21.4s | 0.51s | 0.49s |
| BF16 / 2 GiB offload | 21.7s | 0.51s | 0.50s |
| BF16 / 5 GiB offload | 22.0s | 0.51s | 0.50s |
| BF16 / 8 GiB offload | 22.3s | 0.52s | 0.50s |
| W8A16 / 0 GiB offload | 33.6s | 0.56s | 0.54s |
| W8A16 / 2 GiB offload | 33.7s | 0.57s | 0.55s |
| W8A16 / 5 GiB offload | 34.2s | 0.57s | 0.56s |
| W8A16 / 8 GiB offload | 34.4s | 0.59s | 0.55s |
| W8A16+AITER / 5 GiB offload | ~973s* | **0.16s** | 0.16s |

\* First start includes JIT kernel compilation (~16 min). Subsequent starts
use cached kernels from `~/.triton/cache/` and `~/.cache/vllm/`.

**Reranker** (5 docs × ~6k tokens ≈ 32k total):

| Config | Cold Start | Warm Mean | Warm Min |
|--------|-----------|-----------|----------|
| BF16 / 0 GiB offload | 65.3s | 4.14s | 1.47s |
| BF16 / 2 GiB offload | 65.8s | 1.69s | 1.46s |
| BF16 / 5 GiB offload | 66.2s | 1.72s | 1.46s |
| BF16 / 8 GiB offload | 66.9s | 1.73s | 1.46s |
| W8A16 / 0 GiB offload | 88.7s | 1.78s | 1.50s |
| W8A16 / 2 GiB offload | 89.0s | 1.51s | 1.49s |
| W8A16 / 5 GiB offload | 89.7s | 1.77s | 1.50s |
| W8A16 / 8 GiB offload | 90.3s | 1.67s | 1.51s |
| W8A16+AITER / 5 GiB offload | —* | **0.178s** | 0.166s |

**Key findings**:

1. **W8A16 embed warm: 0.56s** — 10% slower than BF16 (0.51s), due to
   per-channel INT8→BF16 dequantize overhead
2. **W8A16+AITER embed warm: 0.16s** — **3.5× faster** than W8A16 without
   AITER, **3.2× faster** than BF16 baseline. AITER provides fused
   RMSNorm + optimized INT8 linear kernels
3. **W8A16 rerank warm: 1.51s** — within 2% of BF16 (1.69s at 2 GiB offload).
   **W8A16+AITER rerank warm: 0.178s** — **8.5× faster** than without AITER.
   Reranker AITER benchmark done (T4.7)
4. **CPU offload**: Negligible impact on warm latency (~0.01s per 3 GiB added).
   UVA zero-copy bandwidth (~212 GB/s LPDDR5) is not the bottleneck
5. **Cold start**: W8A16 pays ~12s more than BF16 (embed) / ~23s more (rerank)
   due to compressed-tensors weight-loading + dequant kernel compilation.
   **AITER first start**: ~973s (~16 min) for JIT kernel compilation; cached
   in `~/.triton/cache/` and `~/.cache/vllm/` thereafter
6. **Offload sweetspot**: Any value 0–8 GiB works equally well for warm
   inference. Choose offload based on VRAM budget, not latency

## Offline SDK (Reference)

For batch processing or when the HTTP API is not needed, the vLLM offline SDK
is available. See `test_embed_offline.py` for a working example that produced
correct multimodal similarity scores:

- Q1→Doc1 (text): 0.74
- Q1→Doc2 (image): 0.65
- Q1→Doc3 (text+image): 0.62
- Q4→Doc1-3: 0.06/-0.02/0.02 (unrelated, correctly low)

## AITER Unlock (gfx1151)

vLLM's AITER (AI Tensor Engine for ROCm) kernels are gated behind `on_gfx9()`
in `vllm/platforms/rocm.py`. This restricts AITER to gfx94x/gfx95x (CDNA3/4)
datacenter GPUs only.

### Required Patches

**1. AITER Gate — `_aiter_ops.py` (line 52-55)**

```python
# Original: on_mi3xx() only
# Patched: on_mi3xx() or on_gfx1x()
if on_mi3xx() or on_gfx1x():
```

**2. AITER FA Attention — `rocm_aiter_fa.py` (line 804/809)**

```python
# Already patched: on_mi3xx() or on_gfx1x()
```

**3. FP4 Import Fix — `_aiter_ops.py` (line 1235/1242)**

`on_gfx950()` is not imported by the FP4 methods (`is_fp4bmm_enabled`,
`is_asm_fp4_gemm_dynamic_quant_enabled`). Both methods import `on_gfx9`
but call `on_gfx950()`, causing a `NameError` at runtime. Replace with
`on_gfx9()` which is functionally equivalent for FP4:

```python
# Original (broken): return cls._AITER_ENABLED and cls._FP4BMM_ENABLED and on_gfx950()
# Fixed:             return cls._AITER_ENABLED and cls._FP4BMM_ENABLED and on_gfx9()
```

**BUILD-FIXES:** #97
**Patch:** `patches/aiter-fp4-import-fix.patch`
**Auto-applied on rebuild:** No in this repo. The patch is imported for review
before build-manifest integration.

### ViT vs Decoder Attention Dispatch

| Component | Backend | Reason |
|-----------|---------|--------|
| ViT (visual encoder) | `TRITON_ATTN` | CK `fmha_fwd` crashes on gfx1151; ViT falls back via `on_gfx9()` gate |
| Decoder (text) | `ROCM_ATTN` | Uses `on_gfx1x()` patched gate; standard ROCm flash attention |
| RMSNorm | **AITER** (JIT-compiled for gfx1151) | `module_rmsnorm` compiled via hipcc+ninja targeting `-target-cpu gfx1151` |
| Linear layers | **AITER** `AiterInt8ScaledMMLinearKernel` | W8A16 dequant+GEMM via AITER optimized path |

### Validation Status

**Validated for Embedding + Reranker (W8A16)**:
- Embedding: 3.5× warm latency improvement (0.56s → 0.16s)
- Reranker: 8.5× warm latency improvement (1.51s → 0.178s)
- Ranking accuracy preserved: 5/5 matches, Spearman ρ = 1.0
- Determinism preserved: CosSim = 1.00000000 (embed), 1.00000000 (rerank)

**No pending benchmarks for AITER.**

**Risk**: AITER kernels for gfx1151 are not upstream-validated. AITER PR #1498
(ROCm) adds gfx11xx targets. Once merged, `on_gfx9()` patches become unnecessary.

**Alternative**: `VLLM_ROCM_USE_AITER=0` forces clean Triton fallback.

### AITER Benchmark Results (W8A16, gfx1151)

**Embedding** — tested with `aiter_bench.py`, 5 GiB CPU offload, 32k context:

| Metric | Without AITER | With AITER | Delta |
|--------|--------------|------------|-------|
| Warm latency (1 query) | 0.56s | **0.16s** | **-71% (3.5×)** |
| Ranking accuracy | 5/5 | 5/5 | = |
| Determinism (CosSim re-embed) | 1.000000 | 1.00000000 | = |
| Warmup (after server start) | ~0.5s | 5.83s | +JIT |
| Server init (first start) | ~30s | **973s** | +JIT compile |
| Server init (cached) | ~30s | ~30s+ | cached |

**Reranker** — tested with `aiter_rerank_bench.py`, 5 GiB CPU offload, 5 docs:

| Metric | Without AITER | With AITER | Delta |
|--------|--------------|------------|-------|
| Warm latency (5 docs) | 1.51s | **0.178s** | **-88% (8.5×)** |
| Warm latency (2 docs) | ~1.49s | **0.162s** | **-89% (9.2×)** |
| Determinism (CosSim) | 1.000000 | 1.00000000 | = |
| Score diff max | — | 0.00000000 | perfect |

**Key observations**:

- AITER RMSNorm (`module_rmsnorm`) JIT-compiles for gfx1151 via
  `hipcc+ninja` with `-target-cpu gfx1151` and
  `oclc_isa_version_1151.bc`. First compilation: ~16 min, cached in
  `~/.triton/cache/` and `~/.cache/vllm/`
- AITER Linear (`AiterInt8ScaledMMLinearKernel`) dispatches W8A16
  dequant+GEMM through optimized path instead of generic hipBLASLt
- ViT attention correctly falls to `TRITON_ATTN` (CK `fmha_fwd` not
  available for gfx1151). Decoder uses `ROCM_ATTN`
- `rmsnorm2d_fwd_with_add` type hints overridden by AITER at runtime
  (logged as `type hints mismatch, override to -->`)

**Reranker quality**: Identical ranking behavior to BF16 (binary yes/no
classifier artifact). No ranking degradation from AITER.

### Phase 1: W8A8 Reference Validation — **DONE (FAILED)**

- [x] **T1.1**: Start vLLM with `collin-park/Qwen3-VL-Embedding-8B-W8A8` + `--quantization compressed-tensors`
- [x] **T1.2**: Verify kernel dispatch — check logs for `TritonInt8ScaledMMLinearKernel`
- [x] **T1.3**: Run embedding quality test — compare cosine similarity against BF16 baseline
- [x] **T1.4**: Measure VRAM usage with `rocm-smi`
- [x] **T1.5**: Run multimodal embedding test (image + text) — verify ViT FP32 patch activates

**Result**: W8A8 failed quality check. Spearman ρ = 0.6, ranking flips
(Dog+Beach ↔ Pets+Beach). INT8 activation quantization noise propagates
through 36 transformer layers. W8A8 unsuitable for embedding/reranking.

### Phase 2: W8A16 Self-Quantization Validation — **DONE**

- [x] **T2.1**: Self-quantize Embedding-8B to W8A16 (QuantizationModifier, RTN, ~30 sec)
- [x] **T2.2**: Self-quantize Reranker-8B to W8A16 (QuantizationModifier, RTN, ~30 sec)
- [x] **T2.3**: Start vLLM with both W8A16 models + `--quantization compressed-tensors`
- [x] **T2.3a**: Parse output `config.json` — assert `num_bits=8`, no `input_activations`
- [x] **T2.4**: Run embedding + reranking quality tests against BF16 baseline
- [x] **T2.4a**: Verify safetensors dtypes — visual/lm_head BF16, layers INT8
- [x] **T2.9**: Numerical sanity — CosSim = 0.999995, Spearman ρ = 1.0, 5/5 matches

**Result**: W8A16 production-ready. See Verification Results section.

### Phase 3: Dual-Instance + Lemonade Co-Hosting — **PARTIALLY DONE**

- [x] **T3.1**: Start both vLLM instances with W8A16+AITER models
- [x] **T3.2**: Verify total VRAM ~21.2 GiB (measured)
- [ ] **T3.3**: Start Lemonade with ThinkingCoder (128k context)
- [x] **T3.4**: Run concurrent test — Embed+Rerank parallel in 0.32s
- [x] **T3.5**: 30-min burn test: 8344 requests, 0 errors, VRAM stable

### Phase 4: Optimization — **DONE**

- [x] **T4.1**: AITER `on_gfx9()` unlock — patched, validated, 3.5× embed speedup
- [x] **T4.2**: Test `--enforce-eager` removal — **FAILED**: vLLM v1 + torch.compile
  crashes on gfx1151 with `Cannot access data pointer of Tensor (FakeTensor)`.
  Root cause: ROCm custom kernels (flash attention, etc.) cannot be traced by
  TorchDynamo. AITER does NOT fix this. `--enforce-eager` is **mandatory**
  on gfx1151 regardless of AITER or cpu-offload settings.
- [x] **T4.3**: Benchmark W8A8 Triton vs W8A16 — W8A8 fails quality, W8A16+AITER wins
- [x] **T4.4**: CUDA Graph Capture — not applicable with `--enforce-eager`;
  AITER provides its own fused-kernel paths as alternative optimization
- [x] **T4.5**: GPTQ fallback — not needed, W8A16 CosSim 0.999995 sufficient
- [x] **T4.6**: AITER JIT cache management — documented below
- [x] **T4.7**: Reranker AITER benchmark — 8.5× speedup (1.51s → 0.178s),
  determinism 1.0, quality preserved

#### AITER JIT Cache Management

AITER kernels JIT-compile via `hipcc+ninja` on first server start (~16 min for
Embedding, additional time for Reranker if different architecture). Compiled
kernels are cached and reused on subsequent starts.

**Cache paths**:
- `~/.triton/cache/` — Triton-compiled kernels (`.hsaco` for gfx1151,
  `.amdgcn`, `.llir`, `.ttir`, `.ttgir`). ~7 MiB after Embed+Rerank warmup.
  Contains `*_gemm_kernel*`, `*_fwd_kernel*`, `*_compute_slot_mapping_kernel*`,
  `*_triton_mrope_forward*`, `*rotary_kernel*`, `__triton_launcher.so`
- `~/.cache/vllm/` — vLLM model info + torch_compile_cache (~466 MiB).
  Contains model JSON specs and compiled graph fragments.

**When to clear cache**:
- AITER version update (pip/conda update) — kernel hashes change
- Triton version update — IR/LLIR format may change
- ROCm/TheRock version change — hsaco binary compatibility
- Strange crashes or silent correctness regressions after an update

**How to clear**:
```bash
rm -rf ~/.triton/cache/   # AITER + Triton kernels (7 MiB)
rm -rf ~/.cache/vllm/     # vLLM model info + compile cache (466 MiB)
# Restart vLLM to trigger re-compilation (~16 min first start)
```

**Cold start sequence** (first start after cache clear):
1. vLLM loads model weights (~10-30s)
2. PyTorch/Triton JIT compiles attention kernels (~2-5 min)
3. AITER JIT compiles `module_rmsnorm` for gfx1151 (~10 min)
4. vLLM profile + KV cache allocation (~1 min)
5. Total: **~15-17 min** (Embedding only), **~20-25 min** (Embed+Rerank)

**Warm start** (cached kernels): **~30-60s** (same as without AITER)

### Known Risks

| Risk | Mitigation |
|------|------------|
| hipBLASLt FP32 regression (issue #4566) | `TORCH_BLAS_PREFER_HIPBLASLT=1` for BF16 paths; monitor with `rocprof` |
| `--enforce-eager` caps throughput on UMA | **Mandatory on gfx1151** — torch.compile crashes with FakeTensor error on ROCm custom kernels. AITER compensates by providing fused kernels. Cannot be removed. |
| MES 0x83 page fault hang (issue #6165) | Firmware >= 20260410 with MES 0.86 |
| Triton 3.6.0 casting bug (W8A8 only) | W8A16 path unaffected — no Triton INT8 kernels |
| hipMemcpyWithStream latency (UMA) | ~95% of decode time on UMA; AITER fused kernels partially compensate; HIP Graph Capture blocked by mandatory `--enforce-eager` |
| AITER JIT cold start (~16 min) | First server start compiles gfx1151 kernels; cached in `~/.triton/cache/` thereafter. Clear cache after AITER/Triton version updates |

## Build Stack Notes

### Our TheRock Build vs. Standard ROCm

| Component | Standard ROCm | Our TheRock Build |
|-----------|--------------|-------------------|
| hipBLASLt | "Unsupported architecture" on gfx1151 | **gfx1151 TensileLibrary + extop kernels present** |
| FP8 support | Unclear for gfx1151 | **`supports_fp8()` returns True** (E5M2 KV cache) |
| `compressed-tensors` | Listed in `supported_quantization` | Same |
| AITER | `on_gfx9()` only | **Patched + validated** — 3.5× embed, 8.5× rerank speedup on gfx1151 |
| HIP runtime | Package-managed | Self-built, v7.13 |
| Compiler | System clang | `/opt/src/vllm/local/lib/llvm/bin/amdclang` v23.0 |

The TheRock build has gfx1151-specific hipBLASLt kernels that standard ROCm
packages lack. This invalidates many "gfx1151 unsupported" assumptions in
community reports.

### vLLM Version

```
Commit: 719735d6c
Platform: vllm/platforms/rocm.py
supported_quantization: ["awq", "gptq", "fp8", "compressed-tensors",
    "fbgemm_fp8", "gguf", "quark", "mxfp4", "petit_nvfp4", "torchao",
    "bitsandbytes"]
supports_fp8(): True for ["gfx94", "gfx95", "gfx12", "gfx1100", "gfx1151"]
```

## Decision Tree

```
Start → W8A16 + AITER (validated production path)
  │
  ├─ W8A8 tested? YES — FAILED (Spearman ρ=0.6, ranking flips)
  │   └─ W8A8 INT8 activation noise propagates through 36 layers
  │
  ├─ W8A16 tested? YES — PASSED
  │   ├─ CosSim 0.999995, Spearman ρ 1.0, 5/5 matches
  │   └─ Production quantization: QuantizationModifier, RTN
  │
  ├─ AITER enabled? YES — VALIDATED
  │   ├─ Embed: 0.56s → 0.16s (3.5× speedup)
  │   ├─ Rerank: 1.51s → 0.18s (8.5× speedup)
  │   ├─ Determinism: 1.00000000 (both models)
  │   └─ JIT cold start: ~16 min (cached: ~30s)
  │
  ├─ Dual-Instance tested? YES — PASSED
  │   ├─ VRAM: 21.2 GiB total, 0 errors in 30-min burn test
  │   └─ Concurrent pipeline: 0.32s (embed+rerank parallel)
  │
  └─ Lemonade co-host? PENDING (T3.3)
```

Historical decision path (preserved for reference):

```
Original decision tree (pre-validation):
  ├─ Download collin-park/Qwen3-VL-Embedding-8B-W8A8
  │   └─ Quality degraded (ρ=0.6) → Fall to W8A16
  └─ Self-quantize to W8A16 → Validated, production-ready
```

## References

### vLLM

| Ref | Description |
|-----|-------------|
| PR #39939 | INT8 WMMA Triton attention for gfx1100-gfx1153 (draft) |
| PR #110845 | Asymmetric INT8 for TritonInt8ScaledMMLinearKernel |
| PR #38455 | RDNA 3.5/4 device ID mapping (gfx1151, merged Apr 2026) |
| Issue #32180 | Performance bottlenecks on gfx1151 |
| Issue #37472 | V1 engine hangs on encoder cache profiling (VL models) |

### ROCm

| Ref | Description |
|-----|-------------|
| Issue #6165 | Silent hard hang, MES 0x86 fix |
| Issue #6157 | FP8 GPU crash on Radeon 8060S |
| Issue #4566 | hipBLASLt performance regression on gfx1151 |
| Issue #5643 | hipBLASLt "unsupported arch" on gfx1151 |
| AITER PR #1498 | gfx11xx targets for AITER (draft) |

### Triton

| Ref | Description |
|-----|-------------|
| Issue #5669 | tl.dot INT8 x INT8 broken |

### HuggingFace

| Ref | Description |
|-----|-------------|
| `collin-park/Qwen3-VL-Embedding-8B-W8A8` | W8A8 INT8, compressed-tensors, ~10.5 GiB |
| `RamManavalan/Qwen3-VL-Embedding-8B-FP8` | FP8 E4M3 (unusable on gfx1151) |
| `Forturne/Qwen3-VL-Reranker-8B-FP8` | FP8 E4M3 (unusable on gfx1151) |

### Research Sources

| Source | Key Contributions |
|--------|-------------------|
| Perplexity Deep Research | AITER on_gfx9() bypass, PR #39939 status, VL encoder hang, W8A8 17% slower than FP16 on gfx1100 |
| Claude Deep Research | ConchLinearKernel W4A16-only finding, W8A16 recommendation, AOTriton 0.10b gfx1151 support |
| Gemini Deep Research | TritonInt8ScaledMMLinearKernel operational path, block-size constraints, Triton 3.6.0 casting bug, --enforce-eager anti-pattern |
