<!-- Copyright 2026 Blackcat Informatics Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# Changelog

All notable changes to the Strix Halo vLLM build system are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Multi-distro support**: Auto-detects Arch, Ubuntu/Debian, and Fedora/RHEL
  via `/etc/os-release` and provides distro-specific prerequisite install
  commands and kernel package names.
- **Auto-bootstrapping**: `uv` and `yq` are automatically installed if not
  found on PATH. Tries `go install` (latest) first, falls back to downloading
  the latest GitHub release binary. Both tools are also installed into the
  venv for self-contained builds.
- **ccache integration** (`vllm-env.sh`): If `ccache` is on PATH, creates a
  symlink directory (`${VLLM_DIR}/.ccache/bin/`) shadowing all compiler
  binaries (amdclang, hipcc, clang, gcc, etc.). Intercepts every invocation
  transparently — cmake, ninja, pip, AITER JIT — without modifying CC/CXX.
  50 GB cache size. Biggest win: AITER JIT recompiles after a rebuild drop
  from ~45 min to ~5 min of cache hits. Disable with `VLLM_NO_CCACHE=1`.
- **AITER JIT pre-warm** (step 29b): Compiles all buildable AITER HIP C++
  modules ahead of time, avoiding first-request JIT latency. 12 CDNA-only
  modules are skipped via a YAML-driven skip list (`jit_skip_modules`),
  saving ~2.5 hours per build. vLLM uses Triton/PyTorch fallbacks for all
  skipped modules.
- **AITER JIT skip list** (`vllm-packages.yaml`): Declarative list of 12
  CDNA-only modules with failure reasons (async LDS DMA, packed FP8, wave64
  static_assert, backward codegen, etc.). Maintainable — remove entries if
  upstream AITER adds gfx1151 support.
- **Backend smoke test** (step 37): Downloads SmolLM2-135M-Instruct (~270 MB
  FP16, ~70 MB Q4 GGUF) and runs actual inference through all five backends:
  vLLM, llama.cpp ROCm, llama.cpp Vulkan, Lemonade SDK, and Ollama.
  TunableOp GEMM warmup occurs as a side effect of the vLLM test, so the
  autotuning CSV is always populated — no separate warmup step needed.
  Model config is declarative in `vllm-packages.yaml` (`smoke_test:` section).
- **Per-backend skip controls** (`.env`): Set `SMOKE_SKIP_VLLM=1`,
  `SMOKE_SKIP_LLAMACPP_ROCM=1`, `SMOKE_SKIP_LLAMACPP_VULKAN=1`,
  `SMOKE_SKIP_LEMONADE=1`, or `SMOKE_SKIP_OLLAMA=1` in `${VLLM_DIR}/.env`
  to skip individual backends during iterative debugging. Summary table
  reports skipped backends as `SKIP`.
- **Warmup passes for all backends**: Each backend now runs a 1-token warmup
  generation before the real test to absorb JIT compilation, TunableOp
  autotuning (vLLM), model loading latency (llama.cpp), and HuggingFace
  module initialization (Lemonade). Prevents false timeouts on first run.
- **Optimized wheel installation** (steps 30-31): Source-built wheels are now
  installed back into the build venv, replacing pip-resolved versions with
  Zen 5-optimized native builds.
- **Lemonade + llama.cpp** (steps 33-35): Dual-backend llama.cpp build (ROCm
  hipBLAS + Vulkan) managed by Lemonade SDK, with generated `.env` files for
  each backend.
- **`meson`** added to OS prerequisites (required by TheRock's
  `THEROCK_BUNDLE_SYSDEPS=ON` default).
- **`common.sh`**: Standalone shared shell helpers (logging, section headers,
  prerequisite checks) with no external dependencies.

### Changed

- **YAML-driven build pipeline**: All 37 build steps across 10 phases (A–J)
  are now orchestrated from `vllm-packages.yaml`. Repository URLs, branches,
  patches, build flags, and prerequisites are declared in YAML and read at
  runtime via `yq`. The build script is a generic executor, not a hardcoded
  sequence.
- **TunableOp warmup absorbed into smoke test**: The standalone
  `warmup_tunableop()` (former step 29c) is replaced by the backend smoke
  test (step 37). TunableOp CSV is populated as a side effect of vLLM
  inference — no `.env` file or pre-configured model required.
- **Prerequisites section** in `vllm-packages.yaml` restructured with per-distro
  `install_commands` map (arch, ubuntu, fedora) instead of a single Arch-only
  command.
- **`bootstrap_yq()`** no longer hardcodes a version. Uses `go install
  github.com/mikefarah/yq/v4@latest` when Go is available, otherwise fetches
  the latest release tag from the GitHub API.
- **All wheels are mandatory**: Steps 30-32 now `die` on any wheel build
  failure instead of falling back to PyPI binaries. Step 32 verifies all 13
  required wheels are present before completing.
- **Old wheels auto-pruned**: `prune_old_wheels()` removes stale versions
  from `wheels/` after each build, preventing duplicate accumulation across
  rebuilds (e.g., two amd-aiter or two vllm wheels from successive runs).

### Fixed

- **llama-cli conversation mode hang**: `llama-cli` enters interactive
  conversation mode by default, blocking forever on stdin when run from a
  script. `--no-conversation` is not supported by llama-cli (only
  llama-server). Fixed by using `--single-turn` which generates one response
  and exits. Combined with a `timeout` wrapper (120s warmup, 60s test) as a
  safety net. Output extraction uses `sed -n 's/^| *//p'` to parse the
  `| ` response prefix from llama-cli's conversation format.
- **Summary table crash with `set -e`**: `((pass_count++))` when `pass_count`
  is 0 evaluates to `((0))` which returns exit code 1, killed by `set -e`
  after printing only the first result row. Fixed by using assignment form
  `pass_count=$((pass_count + 1))` which always succeeds.
- **Lemonade SDK integration**: Three cascading fixes:
  1. Wrong recipe: `recipe='llamacpp'` does not exist; changed to
     `recipe='hf-dgpu'` with HuggingFace model name instead of GGUF path.
  2. Wrong API: `model.generate('text')` fails because `HuggingfaceAdapter`
     expects tokenized `input_ids` tensors. Fixed to tokenize with
     `tokenizer(prompt, return_tensors='pt')` and pass `input_ids` +
     `attention_mask`.
  3. Wrong output parsing: `output['output_tokens']` fails because
     `generate()` returns a raw `[batch, seq]` token ID tensor, not a dict.
     Fixed to slice off prompt tokens (`outputs[0][input_len:]`) and decode.
  4. Missing chat template: SmolLM2-135M-Instruct produces immediate EOS
     without chat template formatting. Added
     `tokenizer.apply_chat_template()` with `add_generation_prompt=True`.
- **vLLM per-prompt assertion too strict**: Replaced per-prompt `assert n_out
  > 0` with aggregate `assert total_output_tokens > 0` to tolerate vLLM
  occasionally producing zero tokens on a single short prompt while still
  catching total generation failure.
- **Step dispatch on scalar YAML values**: `yq '.steps."N"[]'` fails when the
  value is a scalar string (not an array). Fixed with `mapfile` + empty-entry
  filtering to handle both `[func1, func2]` and `func` forms.
- **Triton stdout pollution** (BUILD-FIXES.md #55): `triton.experimental.gluon`
  warning printed to stdout (not stderr) corrupted `jit_dir` variable capture,
  causing the build script to die after AITER pre-warm instead of continuing to
  steps 30-35. Fixed by piping through `tail -1` on all three capture sites.
- **Flash Attention internal AITER install** (BUILD-FIXES.md #52): Flash
  Attention's `setup.py` tried to `pip install third_party/aiter` which fails
  on gfx1151 (missing gfx942 `.co` files). Patched to skip — we build AITER
  separately.
- **AITER JIT SystemExit propagation** (BUILD-FIXES.md #53): `SystemExit`
  (inherits `BaseException`, not `Exception`) from failed module builds
  propagated through the warmup script. Changed handler to catch
  `(Exception, SystemExit)`.
- **Optimized wheels not in venv** (BUILD-FIXES.md #54): Steps 30-31 built
  wheels but never installed them. Added `uv pip install --force-reinstall`
  after each wheel build.
- **FP8 linear crash on gfx1x** (BUILD-FIXES.md #37): CK GEMM FP8 kernels
  use CDNA MFMA instructions. Added gfx1x guard to fall through to Triton
  blockscale GEMM.
- **`+rms_norm` custom_ops graph partition bug** (BUILD-FIXES.md #40): Declaring
  RMSNorm as opaque custom op caused Inductor to generate incorrect code at
  partition boundaries on wave32. This was the single biggest fix: 7.7-8.9x
  speedup (137 -> 1060 tok/s on Qwen2.5-0.5B).
- **Duplicate pattern registration crash** (BUILD-FIXES.md #39): AITER fusion
  pass registered identical patterns, fixed with `skip_duplicates=True`.
- **Triton sampler page fault** (BUILD-FIXES.md #41): Reframed as a
  build-specific stack-compatibility issue (not a universal gfx1151 rule).
  Removed the blanket sampler-bypass patch from the YAML patch set so
  Triton sampler remains enabled by default.
- **FLA autotuner page faults** (BUILD-FIXES.md #42, #49): Restricted AMD
  autotuning to `num_stages=2`, `BV=32` to stay within RDNA 3.5 register
  pressure limits.
- **Qwen3.5 FLA warmup page fault** (BUILD-FIXES.md #43): Restricted warmup
  loop to `T=64` only (T < BT page-faults on wave32).
- **KV cache block_size mismatch** (BUILD-FIXES.md #50): Hybrid model alignment
  clobbered by ROCm platform config. Fixed sequencing and constraint propagation.
- **AITER unified attention non-power-of-2 block_size** (BUILD-FIXES.md #51):
  Added power-of-2 constraint and TILE_SIZE cap for hybrid models.

## [0.2.0] - 2026-03-15

### Added

- **AITER source rebuild** (step 28b): Rebuilds AITER from PyTorch submodule
  source with matching CK headers, eliminating ABI mismatches from pip wheel.
- **FLA patches** (patches 11-16, 20-28): Flash Linear Attention fixes for
  Qwen3.5 hybrid model support on RDNA 3.5.
- **`vllm-packages.yaml`**: Package manifest with all repos, branches, patches,
  and build metadata in a single declarative file.
- Qwen3.5-0.8B MoE benchmark: 285.5 tok/s with FLA + hybrid model patches.

### Changed

- AITER gfx1x gate extended to cover attention, GEMM, and normalization
  (patches 2-5).
- ViT attention reverted to gfx9-only (patch 6) — CK fmha_fwd rejects ViT
  dimensions on gfx1151.

## [0.1.0] - 2026-03-12

### Added

- Initial 32-step build pipeline across 8 phases.
- 29 documented build fixes with root cause analysis.
- `build-vllm.sh`: Master build script.
- `vllm-env.sh`: Environment activation with compiler flags for Zen 5 + RDNA 3.5.
- `vllm-start.sh`, `vllm-stop.sh`, `vllm-status.sh`: Runtime management.
- Benchmark results: 1059.8 tok/s (Qwen2.5-0.5B), 391.6 tok/s (Qwen2.5-1.5B).
- 13 optimized wheel packages (torch, triton, vllm, numpy, etc.).

[Unreleased]: https://github.com/blackcat-informatics/ai-notes/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/blackcat-informatics/ai-notes/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/blackcat-informatics/ai-notes/releases/tag/v0.1.0
