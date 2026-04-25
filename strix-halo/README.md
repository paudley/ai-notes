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

Ubuntu and Fedora are also supported — the build script detects the distro
and provides the correct package install commands. Any distribution should
work provided the kernel is 7.0+ with the `amdgpu` and `amdxdna` modules
loaded.

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

## Build Pipeline (37 Steps, 10 Phases)

```
Phase A: ROCm SDK (TheRock)
  1. Clone TheRock          4. Build TheRock (~3 hours)
  2. Create bootstrap venv  5. Validate ROCm
  3. Configure TheRock

Phase B: CPU Libraries + Python
  6. Build AOCL-Utils       8. Build Python 3.13 (PGO + LTO)
  7. Build AOCL-LibM        9. Create final venv

Phase C: ML Framework (PyTorch + TorchVision)
 10. Clone PyTorch         13. Clone TorchVision
 11. Build PyTorch (~1-2h) 14. Build TorchVision
 12. Validate PyTorch

Phase D: Kernel Compilers
 15. Clone Triton          18. Clone AOTriton
 16. Build Triton          19. Build AOTriton
 17. Validate Triton

Phase E: Inference Engine
 20. Clone vLLM            24. Install ROCm requirements
 21. Patch amdsmi import   25. Build vLLM (AITER first)
 21b. Patch gfx1151 AITER
 22. Install build deps
 23. use_existing_torch.py

Phase F: Attention (Flash Attention + AITER)
 26. Reinstall amdsmi      29. Build Flash Attention
 27. Clone Flash Attention  29b. Rebuild AITER from source (CK-aligned)
 28. Patch Flash Attention

Phase G: Validation + Warmup
 30. Smoke test
 30b. AITER JIT pre-warm   (compile all buildable modules ahead of time)
 30c. TunableOp warmup     (populate GEMM autotuning CSV)

Phase H: Optimized Wheels (Zen 5 native builds for downstream venvs)
 31. Build Rust wheels     (orjson, cryptography — AVX-512 + VAES)
 32. Build C/C++ wheels    (numpy, sentencepiece, zstandard, asyncpg)
 33. Export source wheels   (torch, triton, torchvision, amd-aiter, amdsmi)

Phase I: Lemonade Inference Server
  34. Clone Lemonade + build llama.cpp (ROCm hipBLAS + Vulkan backends)
  35. Install Lemonade SDK from PyPI
  36. Validate Lemonade (both backends)

Phase J: Backend Smoke Test
  37. Validate all inference backends with SmolLM2
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

## Supported Distributions

The build script auto-detects the distro via `/etc/os-release` and adapts
prerequisite checks and install hints accordingly.

| Family | Distros | Package Manager |
|--------|---------|-----------------|
| **Arch** | CachyOS, Arch, EndeavourOS, Manjaro, Garuda | pacman |
| **Ubuntu** | Ubuntu, Debian, Linux Mint, Pop!_OS, Elementary, Zorin | apt |
| **Fedora** | Fedora, Nobara, RHEL, CentOS, Rocky, Alma | dnf |

`uv` and `yq` are **auto-bootstrapped** if not found on PATH. The script
tries `go install` first (always gets latest), then falls back to
downloading the latest release binary from GitHub. Both tools are also
installed into the managed bootstrap/final venvs for self-contained builds.


The build now creates a **bootstrap virtual environment** from `python3` before
TheRock configuration/build begins, so every early `uv pip install` happens
inside a managed env instead of leaking packages into the system interpreter.
After step 8 compiles the optimized CPython, step 9 recreates the canonical
vLLM environment on `/opt/src/vllm/python/bin/python3`.

## Quick Start

```bash
# 1. Install system prerequisites
#    The build script will tell you exactly what's missing, but here are
#    the full install commands for each distro family:

# Arch / CachyOS:
sudo pacman -S clang lld cmake ninja git curl \
    gcc-fortran patchelf automake libtool bison flex xxd scons meson \
    vulkan-devel vulkan-radeon

# Ubuntu / Debian:
sudo apt install clang lld cmake ninja-build git curl \
    gfortran patchelf automake libtool bison flex xxd scons meson \
    libvulkan-dev mesa-vulkan-drivers

# Fedora / RHEL:
sudo dnf install clang lld cmake ninja-build git curl \
    gcc-gfortran patchelf automake libtool bison flex vim-common scons meson \
    vulkan-devel mesa-vulkan-drivers

# 2. Install a kernel with gfx1151 amdgpu support (kernel 7.0+)
#    Skip if already running kernel 7.0+
#    Arch:   sudo pacman -S linux-cachyos-rc linux-cachyos-rc-headers
#    Ubuntu: Use mainline kernel PPA or HWE kernel >= 7.0
#    Fedora: Rawhide or kernel-next >= 7.0

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
| `build-vllm.sh` | Master build script (37-step pipeline) |
| `vllm-env.sh` | Environment activation (compiler flags, ROCm paths, venv) |
| `vllm-packages.yaml` | Package manifest (repos, branches, patches, per-distro prerequisites, bootstrap config) |
| `vllm-start.sh` | Start all vLLM inference instances (role-based, multi-model) |
| `vllm-stop.sh` | Stop all running vLLM instances (graceful SIGTERM + SIGKILL) |
| `vllm-status.sh` | Check health/PID/model status of all vLLM instances |
| `common.sh` | Shared shell helpers (logging, section headers, prerequisite checks) |
| `vllm-runtime-helpers.sh` | Shared library for start/stop/status scripts |
| `BUILD-FIXES.md` | Detailed documentation of all build patches and workarounds |
| `CHANGELOG.md` | Version history and notable changes |

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
| torch | 647M | C++/HIP |
| triton | 217M | C++/LLVM |
| vllm | 48M | C++/HIP |
| amd-aiter | 43M | C++/HIP |
| numpy | 7.1M | C (meson) |
| cryptography | 2.4M | Rust |
| sentencepiece | 1.5M | C++ (cmake) |
| amdsmi | 1.4M | Pure Python |
| torchvision | 1.3M | C++ |
| zstandard | 940K | C |
| asyncpg | 828K | Cython |
| orjson | 344K | Rust |
| flash_attn | 204K | Pure Python |

All wheels are in `/opt/src/vllm/wheels/` and can be installed into any
Python 3.13 venv.

## Using the Built Wheels

The build produces two key artifacts:

1. **Optimized CPython 3.13** at `/opt/src/vllm/python/bin/python3`
   (PGO + ThinLTO + amdclang `-famd-opt`, Zen 5 native)
2. **13 optimized wheel packages** in `/opt/src/vllm/wheels/`

Both are portable to any environment on the same machine (or any machine
with the same architecture and ROCm libraries at `/opt/src/vllm/local/lib`).

### Quick: pip install into any venv

```bash
# From any activated venv (Python 3.13 required)
pip install /opt/src/vllm/wheels/*.whl
```

### Full Setup: New project with optimized Python + wheels

This is the recommended approach for maximum performance — the optimized
Python interpreter alone provides ~5-15% speedup on compute-bound code.

```bash
# 1. Create a new project with the source-built Python
mkdir my-project && cd my-project
uv venv --python /opt/src/vllm/python/bin/python3 .venv
source .venv/bin/activate

# 2. Install all optimized wheels
uv pip install /opt/src/vllm/wheels/*.whl

# 3. Verify
python -c "import torch; print(f'PyTorch {torch.__version__}, ROCm {torch.version.hip}')"
python -c "import vllm; print(f'vLLM {vllm.__version__}')"
```

### Recommended: uv project with find-links

For uv-managed projects, add to `pyproject.toml` to have uv automatically
resolve source-built wheels from the local directory instead of PyPI:

```toml
[project]
requires-python = ">=3.13"
dependencies = [
    "vllm",
    "torch",
    "numpy",
]

[tool.uv]
find-links = ["/opt/src/vllm/wheels"]
prerelease = "if-necessary-or-explicit"
python-preference = "only-system"

override-dependencies = [
    # Source-built ROCm wheels (dev versions resolved via find-links)
    "torch==2.12.0a0+git7735e5b",
    "triton==3.0.0+gitcb89b617",
    "torchvision==0.26.0a0+5328524",
    "vllm==0.17.1rc1.dev169+g6590a3ecd.d20260315.rocm713",
    "flash-attn==2.8.4",
    "amd-aiter==0.1.11.dev32+g9a469a608.d20260317",
    "amdsmi==26.3.0+093b66caa3.dirty",
    # Zen 5 optimized native wheels
    "numpy==2.4.3",
    "cryptography==46.0.5",
    "orjson==3.11.7",
    "sentencepiece==0.2.1",
    "zstandard==0.25.0",
    "asyncpg==0.31.0",
]
```

Then create the venv with the optimized Python:

```bash
# Point uv at the source-built Python
uv venv --python /opt/src/vllm/python/bin/python3
uv sync
```

**How this works:**
- `find-links` tells uv to check the local wheel directory before PyPI
- `override-dependencies` pins exact versions (including dev/pre-release
  suffixes like `2.12.0a0+git...`) so uv resolves to the local wheels
- `prerelease = "if-necessary-or-explicit"` is needed because source
  builds produce pre-release version strings by default
- `python-preference = "only-system"` prevents uv from downloading a
  generic Python when the optimized one is available

Update the version strings after each rebuild — they change with every
git commit in the upstream repos. To get current versions:

```bash
ls /opt/src/vllm/wheels/*.whl | xargs -I{} basename {} | sed 's/-cp313.*//'
```

### Runtime Environment

The ROCm wheels need the TheRock libraries at runtime. Source
`vllm-env.sh` to set up all paths, or manually set:

```bash
export LD_LIBRARY_PATH="/opt/src/vllm/local/lib:${LD_LIBRARY_PATH}"
export HSA_OVERRIDE_GFX_VERSION=11.5.1    # For gfx1151
export ROCBLAS_USE_HIPBLASLT=1
export VLLM_USE_TRITON_FLASH_ATTN=0       # Use AITER attention
```

Or simply source the activation script:

```bash
source /path/to/strix-halo/vllm-env.sh
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
