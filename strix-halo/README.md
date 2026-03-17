<!-- Copyright 2026 Blackcat Informatics Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# Building vLLM from Source on AMD Strix Halo (gfx1151)

A complete, reproducible reference for compiling the **entire vLLM inference
stack from source** on AMD Strix Halo APUs (Zen 5 + RDNA 3.5, gfx1151).

This is not a guide for installing pip wheels. Every component — from the ROCm
SDK to Python itself — is compiled from source with aggressive optimization
flags targeting the Strix Halo microarchitecture.

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| **CPU** | AMD Zen 5 (Strix Halo / Ryzen AI Max) |
| **GPU** | RDNA 3.5 iGPU, gfx1151, 40 CUs |
| **Memory** | 64-128 GB unified LPDDR5X |
| **Disk** | ~100 GB free for build artifacts |
| **Kernel** | Linux 7.0-rc3+ (amdgpu + amdxdna loaded) |

## Tested Platform

This build is developed and tested on **CachyOS** (Arch-based) with the
**CachyOS kernel 7.0-rc3** (`linux-cachyos-rc`). CachyOS ships patched
kernels with up-to-date amdgpu and amdxdna driver support that gfx1151
requires — mainline kernels prior to 7.0 do not include the necessary
RDNA 3.5 firmware and drm driver changes.

| Component | Version |
|-----------|---------|
| **Distro** | CachyOS (Arch-based, rolling) |
| **Kernel** | 7.0.0-rc3-2-cachyos-rc (`linux-cachyos-rc`) |
| **Compiler** | Clang/LLVM 21+ (system), then amdclang from TheRock |
| **Python** | Built from source (3.13.x, PGO + LTO + amdclang) |
| **ROCm** | Built from source (TheRock nightly) |

Other distributions should work provided the kernel is 7.0+ with the
`amdgpu` and `amdxdna` modules loaded, and the system packages listed
below are available.

## Performance (gfx1151, 40 CUs, unified LPDDR5X)

Benchmarked with FULL CUDA graph capture, ALL AITER optimizations
(attention, GEMM, normalization), torch.compile, hipBLASLt, and TunableOp:

| Model | Parameters | tok/s | Configuration |
|-------|-----------|-------|---------------|
| Qwen2.5-0.5B-Instruct | 494M | 1059.8 | FULL graph + ALL AITER |
| Qwen2.5-1.5B-Instruct | 1.5B | 391.6 | FULL graph + ALL AITER |
| Qwen3.5-0.8B (MoE) | 0.8B active / 3B total | 285.5 | enforce_eager (FLA + hybrid model patches, see BUILD-FIXES.md #42-43, #49-51) |

These numbers represent steady-state decode throughput (8 concurrent
prompts, 128 max tokens, after 2 warmup passes). The build patches in
this repository are required — without them, gfx1151 is limited to
~137 tok/s (0.5B) / ~44 tok/s (1.5B) due to Inductor codegen bugs
that force PIECEWISE graph mode with AITER disabled.

## Why Build from Source?

1. **gfx1151 is not in upstream ROCm** (as of ROCm 6.x / early 7.x).
   TheRock ROCm nightly is the only way to get a working HIP toolchain.

2. **Unified memory architecture** means CPU and GPU share LPDDR5X.
   Cache locality optimizations (Polly, `-mprefer-vector-width=512`)
   matter more than on discrete GPUs.

3. **Native AVX-512** on Zen 5 without the clock penalty of Zen 4.
   Source builds with `-march=native` unlock full 512-bit execution.

4. **AMD-specific compiler optimizations** via amdclang's `-famd-opt`
   flag (proprietary Zen microarchitecture tuning not available in
   upstream LLVM).

## Build Pipeline (35 Steps, 9 Phases)

```
Phase A: ROCm SDK (TheRock)
  1. Clone TheRock          3. Build TheRock (~3 hours)
  2. Configure TheRock      4. Validate ROCm

Phase B: CPU Libraries + Python
  5. Build AOCL-Utils       7. Build Python 3.13 (PGO + LTO)
  6. Build AOCL-LibM        8. Create venv

Phase C: ML Framework (PyTorch + TorchVision)
  9. Clone PyTorch         12. Clone TorchVision
 10. Build PyTorch (~1-2h) 13. Build TorchVision
 11. Validate PyTorch

Phase D: Kernel Compilers
 14. Clone Triton          17. Clone AOTriton
 15. Build Triton          18. Build AOTriton
 16. Validate Triton

Phase E: Inference Engine
 19. Clone vLLM            23. Install ROCm requirements
 20. Patch amdsmi import   24. Build vLLM (AITER first)
 20b. Patch gfx1151 AITER
 21. Install build deps
 22. use_existing_torch.py

Phase F: Attention (Flash Attention + AITER)
 25. Reinstall amdsmi      28. Build Flash Attention
 26. Clone Flash Attention  28b. Rebuild AITER from source (CK-aligned)
 27. Patch Flash Attention

Phase G: Validation + Warmup
 29. Smoke test
 29b. AITER JIT pre-warm   (compile all buildable modules ahead of time)
 29c. TunableOp warmup     (populate GEMM autotuning CSV)

Phase H: Optimized Wheels (Zen 5 native builds for downstream venvs)
 30. Build Rust wheels     (orjson, cryptography — AVX-512 + VAES)
 31. Build C/C++ wheels    (numpy, sentencepiece, zstandard, asyncpg)
 32. Export source wheels   (torch, triton, torchvision, amd-aiter, amdsmi)

Phase I: Lemonade Inference Server
 33. Clone Lemonade + build llama.cpp (ROCm hipBLAS + Vulkan backends)
 34. Install Lemonade SDK from PyPI
 35. Validate Lemonade (both backends)
```

### Lemonade: Dual-Backend llama.cpp

Phase I builds [llama.cpp](https://github.com/ggml-org/llama.cpp) with
two GPU backends, managed by the
[Lemonade SDK](https://pypi.org/project/lemonade-sdk/):

| Backend | Best For | Notes |
|---------|----------|-------|
| **ROCm** (hipBLAS) | Prefill < 32K context | Primary backend, uses amdclang + gfx1151 HIP flags |
| **Vulkan** | Generation speed, prefill > 32K | +22% tok/s generation, no 32K VMM limitation |

Both backends are installed into the venv and Lemonade can route between
them based on workload. Each backend gets its own `.env` file with
gfx1151 runtime optimizations (batch sizing, hipBLASLt, THP).

## Quick Start

```bash
# 1. Install system prerequisites (CachyOS / Arch Linux)
sudo pacman -S clang lld cmake ninja git curl uv \
    gcc-fortran patchelf automake libtool bison flex xxd scons \
    vulkan-devel vulkan-radeon   # For Vulkan llama.cpp backend

# 2. Install CachyOS RC kernel (for gfx1151 amdgpu support)
#    Skip if already running kernel 7.0+
sudo pacman -S linux-cachyos-rc linux-cachyos-rc-headers

# 3. Create the build directory (all source lives under /opt/src/vllm)
sudo mkdir -p /opt/src/vllm
sudo chown $(id -u):$(id -g) /opt/src/vllm

# 4. Run the full build
./build-vllm.sh

# 5. Activate the environment (for interactive use)
source ./vllm-env.sh
source ./vllm-env.sh --info   # Show settings
```

### Resuming and Rebuilding

```bash
./build-vllm.sh --step 14   # Resume from step 14 (e.g., after Triton fix)
./build-vllm.sh --rebuild    # Clean everything and rebuild from scratch
```

## Compiler Flags

### CPU (Zen 5) -- CFLAGS/CXXFLAGS

```
-O3 -march=native -flto=thin -mprefer-vector-width=512
-mavx512f -mavx512dq -mavx512vl -mavx512bw
-famd-opt                          # amdclang only: Zen microarch tuning
-mllvm -polly                      # polyhedral loop optimizer
-mllvm -polly-vectorizer=stripmine # cache hierarchy restructuring
-mllvm -inline-threshold=600       # aggressive inlining for wide pipeline
-mllvm -unroll-threshold=150       # aggressive unrolling for large ROB
-mllvm -adce-remove-loops          # dead loop cleanup
```

### GPU (RDNA 3.5, gfx1151) -- HIP_CLANG_FLAGS

```
--offload-arch=gfx1151
-mllvm -amdgpu-early-inline-all=true     # keep VALU busy
-mllvm -amdgpu-function-calls=false      # eliminate call overhead on iGPU
-famd-opt
```

### Rust -- RUSTFLAGS

```
-C target-cpu=znver5 -C opt-level=3
```

Explicit `znver5` (not `native`) because Rust's native detection has a bug
where it identifies znver5 but only enables SSE2. Explicit znver5 enables
all 40+ target features including AVX-512, VAES, VPCLMULQDQ, GFNI, SHA.

## Files

| File | Description |
|------|-------------|
| `build-vllm.sh` | Master build script (35-step pipeline) |
| `vllm-env.sh` | Environment activation (compiler flags, ROCm paths, venv) |
| `vllm-packages.yaml` | Package manifest (repos, branches, patches, build metadata) |
| `vllm-start.sh` | Start all vLLM inference instances (role-based, multi-model) |
| `vllm-stop.sh` | Stop all running vLLM instances (graceful SIGTERM + SIGKILL) |
| `vllm-status.sh` | Check health/PID/model status of all vLLM instances |
| `common.sh` | Shared shell helpers (logging, section headers, prerequisite checks) |
| `vllm-runtime-helpers.sh` | Shared library for start/stop/status scripts |
| `BUILD-FIXES.md` | Detailed documentation of all build patches and workarounds |

## Repo Variants

| Component | Repository | Branch |
|-----------|-----------|--------|
| TheRock | ROCm/TheRock | main |
| PyTorch | ROCm/pytorch | develop |
| TorchVision | pytorch/vision | main |
| Triton | ROCm/triton | main_perf |
| Flash Attention | ROCm/flash-attention | main_perf |
| vLLM | vllm-project/vllm | main |
| AOTriton | ROCm/aotriton | main |
| AOCL-LibM | amd/aocl-libm-ose | main |
| AOCL-Utils | amd/aocl-utils | main |
| llama.cpp | ggml-org/llama.cpp | master |
| Lemonade | lemonade-sdk/lemonade | v10.0.0 |

Note: PyTorch, Triton, and Flash Attention use the **ROCm forks**, not
upstream. The ROCm forks carry AMD-specific fixes (hipify patches, Tensile
integration, rocm_smi linkage) that haven't been upstreamed. vLLM uses
upstream (the ROCm fork is deprecated).

## Output

When complete, the build produces 13 optimized wheel packages:

| Wheel | Size | Type |
|-------|------|------|
| torch | 681M | C++/HIP |
| triton | 227M | C++/LLVM |
| vllm | 50M | C++/HIP |
| torchvision | ~30M | C++ |
| numpy | 7.4M | C (meson) |
| cryptography | 2.4M | Rust |
| sentencepiece | 1.5M | C++ (cmake) |
| amdsmi | 1.4M | Pure Python |
| zstandard | 961K | C |
| asyncpg | 845K | Cython |
| orjson | 349K | Rust |
| flash_attn | 206K | Pure Python |
| amd-aiter | ~2M | C++/HIP |

All wheels are in `/opt/src/vllm/wheels/` and can be installed into any
Python 3.13 venv.

## Using the Built Wheels

### Quick: pip

```bash
pip install /opt/src/vllm/wheels/*.whl
```

### Recommended: uv with find-links

Add to your project's `pyproject.toml` to have uv automatically resolve
source-built wheels from the local directory instead of PyPI:

```toml
[tool.uv]
find-links = ["/opt/src/vllm/wheels"]
prerelease = "if-necessary-or-explicit"

override-dependencies = [
    # Source-built ROCm wheels (dev versions resolved via find-links)
    "torch==2.12.0a0+git7735e5b",
    "triton==3.0.0+gitcb89b617",
    "torchvision==0.26.0a0+5328524",
    "vllm==0.17.1rc1.dev169+g6590a3ecd.d20260315.rocm713",
    "flash-attn==2.8.4",
    "amd-aiter==0.1.0+gitabcdef",
    "amdsmi==26.3.0+093b66caa3.dirty",
    # Zen 5 optimized native wheels
    "numpy==2.4.3",
]
```

The `find-links` directive tells uv to check the local wheel directory
before PyPI. The `override-dependencies` pins exact versions (including
dev/pre-release suffixes like `2.12.0a0+git...`) so uv resolves to the
local wheels. The `prerelease` setting is needed because source builds
produce pre-release version strings by default.

Update the version strings after each rebuild — they change with every
git commit in the upstream repos.

### Using the Source-Built Python

The build produces an optimized CPython 3.13 at
`/opt/src/vllm/python/bin/python3` (PGO + ThinLTO + amdclang). To use
it with uv:

```bash
# Create a venv using the source-built Python
uv venv --python /opt/src/vllm/python/bin/python3 .venv
source .venv/bin/activate

# Install all built wheels
uv pip install /opt/src/vllm/wheels/*.whl
```

## Runtime Management

The `vllm-start.sh`, `vllm-stop.sh`, and `vllm-status.sh` scripts
manage multiple vLLM inference instances via a role-based configuration
system. Each role (e.g., `director`, `voice`) gets its own model, port,
device assignment, and GPU memory allocation.

### Configuration

Create a `.env` file with role definitions:

```bash
# Roles to launch (space-separated)
VLLM_ROLES="director voice"

# Per-role configuration
VLLM_DIRECTOR_MODEL="Qwen/Qwen3-32B"
VLLM_DIRECTOR_PORT=8100
VLLM_DIRECTOR_DEVICE=rocm
VLLM_DIRECTOR_GPU_MEMORY_MB=40960

VLLM_VOICE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
VLLM_VOICE_PORT=8101
VLLM_VOICE_DEVICE=rocm
VLLM_VOICE_GPU_MEMORY_MB=16384
```

### Usage

```bash
source ./vllm-env.sh       # Activate the vLLM environment
./vllm-start.sh            # Start all roles
./vllm-status.sh           # Check health + loaded models
./vllm-stop.sh             # Graceful shutdown
```

## License

Copyright 2026 Blackcat Informatics Inc.

[MIT](../LICENSE) — The upstream projects (TheRock, PyTorch, Triton,
vLLM, etc.) each have their own licenses. See the respective
repositories for details.
