# ai-notes

Notes, scripts, and solutions for running local AI models on AMD hardware.

Covers optimization issues, build workarounds, and practical solutions
discovered while building inference stacks from source on bleeding-edge
silicon.

## Contents

### [strix-halo/](strix-halo/)

Complete from-source build pipeline for the **vLLM inference stack** on
AMD Strix Halo APUs (Zen 5 + RDNA 3.5, gfx1151). Compiles every component
-- from the ROCm SDK to Python itself -- with aggressive optimization flags
targeting the Strix Halo unified memory architecture.

- **40-step build pipeline** across 11 phases (TheRock ROCm → AOCL → Python → PyTorch + TorchVision → Triton → vLLM → Flash Attention → optimized wheels → Lemonade → backend smoke test + benchmarks → deferred AITER JIT pre-warm)
- **55+ documented build fixes** with root cause analysis ([BUILD-FIXES.md](strix-halo/BUILD-FIXES.md))
- **Environment activation script** with compiler flags for Zen 5 CPU + RDNA 3.5 GPU
- Native AVX-512, Polly loop optimizer, AMD-specific `-famd-opt` tuning
- Strict local toolchain policy: downstream optimized phases require TheRock
  `amdclang`, source-built CPython, local ROCm math/runtime libraries
  including hipBLASLt, hipSPARSELt, MIOpen, RCCL, rocBLAS/rocSPARSE, local
  TheRock `libomp`, and fail immediately on missing artifacts, `/opt/rocm`
  leakage, or system OpenMP fallback.
- Builds 28+ optimized Python wheels, including Rust/native server hot paths
  such as `tokenizers`, `safetensors`, `pydantic-core`, `uvloop`,
  `httptools`, `msgspec`, `aiohttp`, and `watchfiles`.
- Current full-stack smoke coverage verifies source-built CPython 3.13.12,
  PyTorch 2.12.0a0, upstream vLLM 0.19.2rc1 dev, TheRock ROCm 7.13, local
  AOTriton/AOCL-LibM, flash attention import, and gfx1151 AITER enablement.

## Hardware

Primary target is the AMD Strix Halo platform:

| Component | Specification |
|-----------|--------------|
| **CPU** | Zen 5, 16 cores, native 512-bit AVX-512 |
| **GPU** | RDNA 3.5 iGPU, gfx1151, 40 CUs |
| **Memory** | 128 GB unified LPDDR5X (shared CPU/GPU) |
| **NPU** | XDNA 2 (Phoenix) |

## License

[MIT](LICENSE)

The upstream projects referenced by these scripts (TheRock, PyTorch, Triton,
vLLM, etc.) each have their own licenses. See the respective repositories
for details.
