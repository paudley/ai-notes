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

- **36-step build pipeline** across 10 phases (TheRock ROCm → AOCL → Python → PyTorch + TorchVision → Triton → vLLM → Flash Attention → optimized wheels → Lemonade → backend smoke test)
- **55+ documented build fixes** with root cause analysis ([BUILD-FIXES.md](strix-halo/BUILD-FIXES.md))
- **Environment activation script** with compiler flags for Zen 5 CPU + RDNA 3.5 GPU
- Native AVX-512, Polly loop optimizer, AMD-specific `-famd-opt` tuning

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
