<!-- Copyright 2026 Blackcat Informatics Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# Build Fixes and Workarounds

Every patch applied by `build-vllm.sh`, documented with root cause analysis
and rationale. These are the real-world issues you'll hit building the vLLM
stack from source on a bleeding-edge AMD platform with modern compilers.

## TheRock ROCm SDK (Phase A)

### 1. elfutils -Werror

**Symptom**: Build fails in elfutils with implicit `const void*` -> `struct*`
conversion errors.

**Root cause**: elfutils' `config/eu.am` unconditionally adds `-Werror` to
every compile command. Modern compilers (Clang 21+, GCC 15+) reject the
implicit pointer conversions from `bsearch()` return values that older
compilers accepted.

**Fix**: Inject `CFLAGS=-Wno-error` into the elfutils CMakeLists.txt
`./configure` environment, effectively neutralizing `-Werror`.

```bash
sed -i 's|"CPPFLAGS=${EXTRA_CPPFLAGS}"|"CPPFLAGS=${EXTRA_CPPFLAGS}"\n      "CFLAGS=-Wno-error"|' \
    third-party/sysdeps/linux/elfutils/CMakeLists.txt
```

### 2. rocprofiler-sdk vendored yaml-cpp missing `<cstdint>`

**Symptom**: `uint16_t`/`uint32_t` undeclared in `emitterutils.cpp`.

**Root cause**: Newer compilers (Clang 18+, GCC 13+) removed transitive
includes of `<cstdint>`. The vendored yaml-cpp relied on getting `uint16_t`
through other headers.

**Fix**: Add explicit `#include <cstdint>` to `emitterutils.cpp`.

### 3. rocprofiler-sdk vendored elfio missing `<cstdint>`

**Symptom**: `Elf64_Half` (typedef for `uint16_t`) undeclared in `elf_types.hpp`.

**Root cause**: Same as #2 -- transitive include removal in modern compilers.

**Fix**: Add `#include <cstdint>` after the header guard in `elf_types.hpp`.

### 4. Polly not enabled by default

**Symptom**: amdclang doesn't support `-mllvm -polly` after TheRock build.

**Root cause**: TheRock's `LLVM_ENABLE_PROJECTS` list doesn't include `polly`.

**Fix**: Patch `compiler/pre_hook_amd-llvm.cmake` to add `polly` to the
semicolon-separated project list:

```bash
sed -i 's|clang;lld;clang-tools-extra;flang|clang;lld;clang-tools-extra;flang;polly|' \
    compiler/pre_hook_amd-llvm.cmake
```

### 5. TheRock requires GCC

**Symptom**: Build fails with "GNU compiler required" errors from
rocprofiler-systems.

**Root cause**: rocprofiler-systems has an explicit GNU compiler check that
blocks Clang. TheRock's internal LLVM build also expects GCC as the host
compiler.

**Fix**: Configure TheRock with `-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++`.
Use GCC only for TheRock; all downstream builds use amdclang from TheRock.

### 6. CachyOS system CFLAGS contamination

**Symptom**: TheRock build fails with unknown flag errors like `-fipa-pta`,
`-fvect-cost-model=very-cheap`, `-flto=20`.

**Root cause**: CachyOS (and some other Arch-based distros) set aggressive
GCC-specific CFLAGS/CXXFLAGS/LDFLAGS in the system environment. These flags
are invalid for clang and break any build that inherits them.

**Fix**: Unset `CFLAGS`, `CXXFLAGS`, `LDFLAGS` before TheRock configure and
build steps. Re-source `vllm-env.sh` afterward to restore our amdclang flags.

### 7. HSA_OVERRIDE_GFX_VERSION leak

**Symptom**: HIPCC fails with `--offload-arch=Invalid` during TheRock build.

**Root cause**: `vllm-env.sh` sets `HSA_OVERRIDE_GFX_VERSION=11.5.1` which
leaks into HIPCC's arch detection, producing invalid target strings.

**Fix**: Unset `HSA_OVERRIDE_GFX_VERSION` during TheRock build.

### 8. Tensile Python dependencies

**Symptom**: hipBLASLt build fails with `ModuleNotFoundError: No module
named 'joblib'` (or msgpack, numpy, pandas).

**Root cause**: hipBLASLt's Tensile kernel generator imports these packages
at code-generation time. They're not declared as cmake dependencies.

**Fix**: `uv pip install joblib msgpack numpy pandas pyyaml pytest` before
the TheRock build.

### 9. MLIR object libraries not installed

**Symptom**: Triton build fails looking for MLIR CAPI object libraries.

**Root cause**: TheRock's `cmake --install` copies `.a` and `.so` files but
skips the `objects-Release/` directory that MLIR's cmake exports reference.

**Fix**: Manually copy `objects-Release/` from the build tree to the install
tree after `cmake --install`.

### 10. FileCheck not installed

**Symptom**: Triton's cmake fails looking for `FileCheck` binary.

**Root cause**: TheRock builds FileCheck (an LLVM test utility) but doesn't
install it when `THEROCK_BUILD_TESTING=OFF`.

**Fix**: Copy `FileCheck` from the build tree to the LLVM bin directory.

## AOCL-LibM (Phase B, Step 6)

### 11. -muse-unaligned-vector-move (AOCC-only flag)

**Symptom**: Build fails with "unknown argument: '-muse-unaligned-vector-move'"

**Root cause**: AOCL-LibM's SConscript assumes any clang >= 14.0.6 is AOCC
(AMD's proprietary compiler). TheRock's open-source amdclang doesn't support
this AOCC-specific flag.

**Fix**: Patch the SConscript to remove the flag injection.

### 12. CMPLX/CMPLXF macro redefinition with -Werror

**Symptom**: Fatal error from `-Werror` + `-Wmacro-redefined` on CMPLX macros.

**Root cause**: AOCL-LibM headers redefine macros that glibc's `complex.h`
already provides (identically).

**Fix**: Add `-Wno-macro-redefined` after `-Werror` in the SConscript.

### 13. Linker flags incompatible with clang

**Symptom**: Link fails with `-ealm_main` not recognized.

**Root cause**: GCC passes `-e` directly to the linker; clang requires
`-Wl,-e,alm_main` syntax. Additionally, AOCL-LibM's hand-written AVX
assembly uses absolute relocations that lld rejects in shared libraries.

**Fix**: Change `-ealm_main` to `-Wl,-e,alm_main` and add `-fuse-ld=bfd`
to use GNU ld for the final link (GNU ld handles the absolute relocations
via dynamic text relocations).

## AOCL-Utils (Phase B, Step 5)

### 14. No LTO for AOCL-Utils

**Why**: AOCL-LibM links against AOCL-Utils `.a` files using GNU ld (see #13).
GNU ld cannot read LLVM bitcode objects produced by `-flto=thin`. Build
AOCL-Utils without LTO.

### 15. clang-tidy crashes

**Symptom**: AOCL-Utils build hangs or crashes during clang-tidy analysis.

**Root cause**: AOCL-Utils auto-enables clang-tidy if found on PATH. Both
TheRock's clang-tidy (crashes on cleanup) and system clang-tidy (doesn't
understand `-famd-opt`) cause failures.

**Fix**: Set `CMAKE_CXX_CLANG_TIDY="/bin/true"` to satisfy the cmake check
while doing nothing.

## CPython (Phase B, Step 7)

### 16. AOCL-LibM breaks PGO test_math

**Symptom**: CPython's PGO profiling run fails on `test_math`.

**Root cause**: AOCL-LibM's transcendentals have slightly different ULP
rounding than glibc's libm:
- `cbrt(-0.0)` returns `+0.0` (should return `-0.0`)
- `fmod(-10, 1)` returns incorrect results
- `nextafter()` and `ulp()` are broken

**Fix**: Do NOT link CPython against `-lalm`. Unset all vllm-env.sh env vars
(which include `-lalm` in LDFLAGS) before CPython's `./configure`. AOCL-LibM
is available at runtime via `LD_LIBRARY_PATH` for downstream libraries
(NumPy, PyTorch) that benefit from it but don't run CPython's exact math tests.

## PyTorch (Phase C, Step 10)

### 17. HIPGraph.hip cudaGraphConditionalHandle

**Symptom**: Compilation error on undefined `cudaGraphConditionalHandle` type.

**Root cause**: PyTorch's hipify step creates `HIPGraph.hip` containing a
`set_conditional_handle()` function that references `cudaGraphConditionalHandle`
-- a CUDA 12.4+ type with no HIP equivalent. Dead code that fails to compile.

**Fix**: Replace `HIPGraph.hip` with a minimal stub containing only the
namespace declaration and a comment explaining the removal.

### 18. -fclang-abi-compat=17 ABI mismatch

**Symptom**: Undefined symbol errors at link time (e.g., `const_data_ptr<Half>`
mangled differently between `libtorch_cpu.so` and `libtorch_hip.so`).

**Root cause**: PyTorch's `cmake/Dependencies.cmake` adds
`-fclang-abi-compat=17` to HIPCC flags "for compat with newer hip-clang C++20
mangling rules". This forces HIP device code to use Clang 17 ABI while host
code uses amdclang 22 ABI, causing name mangling mismatches.

**Fix**: Remove the `-fclang-abi-compat=17` line from `Dependencies.cmake`.

### 19. Missing librocm_smi64.so linkage (upstream bug)

**Symptom**: `undefined symbol: rsmi_init` at runtime when loading PyTorch.

**Root cause**: PyTorch's hipify maps `nvml.h` -> `rocm_smi/rocm_smi.h` in
headers, so `rsmi_*` functions are compiled into `libtorch_hip.so`. But the
build system never adds `-lrocm_smi64` to the link line.

**Fix**: Post-build `patchelf --add-needed librocm_smi64.so libtorch_hip.so`.
This is a real upstream PyTorch bug worth reporting.

### 20. Google Benchmark -Werror with C2y extensions

**Symptom**: Build fails on `__COUNTER__` flagged as C2y extension.

**Root cause**: amdclang 22 flags `__COUNTER__` as a C2y extension in C++
mode. Google Benchmark uses `-Werror`, making this fatal.

**Fix**: `BUILD_TEST=0 USE_BENCHMARK=0` (not needed for inference).

## Triton (Phase D, Step 15)

### 21. -Werror + _POSIX_C_SOURCE redefinition

**Symptom**: Build fails with `-Wmacro-redefined` on `_POSIX_C_SOURCE`.

**Root cause**: Triton's CMakeLists.txt hardcodes `-Werror`. Our custom-built
Python 3.13's `pyconfig.h` redefines `_POSIX_C_SOURCE`, triggering
`-Wmacro-redefined` which becomes fatal with `-Werror`.

**Fix**: Remove `-Werror` from Triton's CMakeLists.txt.

### 22. ROCm/triton setup.py location

**Symptom**: `pip wheel .` fails with "no setup.py found".

**Root cause**: ROCm's Triton fork keeps `setup.py` in `python/` subdirectory
(upstream moved it to repo root).

**Fix**: Detect and use `python/` subdirectory if `setup.py` is there.

## AOTriton (Phase D, Step 18)

### 23. Stray git rebase "pick" line

**Symptom**: cmake parse error on `pick <hash>` line in CMakeLists.txt.

**Root cause**: Upstream ROCm/aotriton main has a stray git rebase "pick"
line at the end of CMakeLists.txt (accidentally committed).

**Fix**: `sed -i '/^pick /d' CMakeLists.txt`

## PyTorch (Phase C, Step 10)

### 21. numpy>=2 ABI compatibility

**Symptom**: Runtime `ImportError` or ABI mismatch when numpy 2.x is installed
alongside a PyTorch wheel built against numpy 1.x headers.

**Root cause**: numpy 2.0 changed C header locations from `numpy/core/include`
to `numpy/_core/include` and introduced ABI changes. Building PyTorch against
`numpy<2` then installing `numpy>=2` at runtime causes header mismatches and
potential segfaults in C extensions.

**Fix**: Build PyTorch (and all downstream) against `numpy>=2.0,<3`. This
ensures the wheel's compiled extensions are ABI-compatible with numpy 2.x
at runtime. The old `numpy<2` downgrade guard in `install_rocm_requirements`
is no longer needed and was removed.

### 22. PyTorch .so patches not baked into wheel

**Symptom**: PyTorch wheel installed on a different machine lacks RPATH fixes
and `librocm_smi64.so` NEEDED entry, causing runtime `undefined symbol: rsmi_init`.

**Root cause**: The original build flow ran `pip wheel .` (which compiles AND
packages in one step), then patched .so files with `patchelf`. But the wheel
was already written to disk -- the patches only affected the local build tree,
not the distributable wheel.

**Fix**: Build the wheel normally with `pip wheel .`, then unpack the `.whl`,
apply patchelf fixes to the `.so` files inside, and repack with Python's
`zipfile` module. This is the only reliable approach because `pip wheel .`
re-invokes cmake which copies fresh `.so` files, overwriting any pre-build
patches.

```bash
# 1. Build wheel normally
pip wheel . --no-build-isolation --no-deps --wheel-dir $WHEELS_DIR -v
# 2. Unpack → patch → repack
unzip -q "$WHEEL" -d "$TMPDIR"
patchelf --set-rpath '$ORIGIN:/opt/src/vllm/local/lib' torch/lib/libtorch_python.so
patchelf --add-needed librocm_smi64.so torch/lib/libtorch_hip.so
# repack with zipfile
```

### 22b. Build tree RPATH leak in libtorch_python.so

**Symptom**: `import torch` fails with `undefined symbol: rsmi_init` pointing
at `/opt/src/vllm/pytorch/build/lib/libtorch_hip.so` even though the installed
wheel copy has the fix.

**Root cause**: cmake bakes the build tree path
(`/opt/src/vllm/pytorch/build/lib`) into `libtorch_python.so`'s RUNPATH. When
the dynamic linker loads `libtorch_hip.so`, it finds the unpatched build-tree
copy before the installed (patched) copy.

**Fix**: During the unpack/patch/repack step, clean all `.so` RPATHs to remove
build tree references:

```bash
patchelf --set-rpath '/opt/src/vllm/local/lib:$ORIGIN' torch/lib/libtorch_python.so
```

### 22c. NumPy 2.0 ABI target version

**Symptom**: `import torch` crashes with "A module that was compiled using
NumPy 1.x cannot be run in NumPy 2.4.3".

**Root cause**: PyTorch's `numpy_stub.h` does NOT set `NPY_TARGET_VERSION`.
Without this, numpy 2.x headers compile against the oldest compatible API
(1.20 / `0x0e`) by default, producing a `.so` that fails the runtime ABI
check when loaded with numpy >= 2.0.

**Fix**: Patch `torch/csrc/utils/numpy_stub.h` to set `NPY_TARGET_VERSION`
before including `<numpy/arrayobject.h>`:

```c
#ifndef NPY_TARGET_VERSION
#define NPY_TARGET_VERSION 0x00000012  /* NPY_2_0_API_VERSION */
#endif
```

The hex value must be used directly because `NPY_2_0_API_VERSION` is defined
inside `numpyconfig.h` which is included by `arrayobject.h`.

## TorchVision (Phase C, Steps 12-13)

### 23. TorchVision source build

**Non-issue**: TorchVision is now built from source (steps 12-13) against
the source-built PyTorch to ensure ABI compatibility. Uses amdclang from
TheRock. CPU-only (no CUDA/ROCm GPU ops -- TorchVision's GPU kernels are
not needed for inference).

## Triton (Phase D, Step 15)

Note: Triton steps renumbered from 12-14 to 14-16 due to TorchVision insertion.

## Wheel Builds (Phase H)

### 24. cmake pip wrapper in build isolation

**Symptom**: sentencepiece source build fails with `ImportError: No module
named 'cmake'`.

**Root cause**: The `cmake` pip package installs a Python wrapper at
`.venv/bin/cmake` that does `from cmake import cmake`. Inside pip's build
isolation, the cmake Python module isn't available, so the wrapper fails.

**Fix**: Replace the Python wrapper with a symlink to the real system cmake
binary (`/usr/bin/cmake`).

### 25. meson -Werror vs -mllvm flags (numpy build)

**Symptom**: numpy build fails on meson capability probes.

**Root cause**: meson hard-codes `-Werror=unused-command-line-argument` in
`ClangCompiler.get_compiler_check_args()` AFTER our CFLAGS. Driver-level
`-mllvm` flags are reported as "unused" in compile-only checks (`-c`),
killing every meson capability probe.

**Fix**: Transform `-mllvm <arg>` to `-Xclang -mllvm -Xclang <arg>` in
CFLAGS for wheel builds. `-Xclang` passes flags directly to the compiler
frontend/backend, bypassing the driver's argument tracking. Move `-famd-opt`
to LDFLAGS (link-time only, no-op at compile time).

### 26. Rust + amdclang linker

**Symptom**: Cargo fails to link with "cc: error: unrecognized argument".

**Root cause**: AMD's `cc` symlink (created in step 4) rejects binaries
not prefixed by "amd". Cargo uses `cc` by default.

**Fix**: Set `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=amdclang`.
Also unset CFLAGS/CXXFLAGS/LDFLAGS because they contain clang-specific
flags that Rust's internal `cc` invocations don't understand.

### 27. Rust -C target-cpu=native bug

**Symptom**: Rust binary only uses SSE2 despite running on Zen 5.

**Root cause**: Rust's native CPU detection identifies znver5 but only
enables SSE2 features (a known `rustc` bug).

**Fix**: Use explicit `-C target-cpu=znver5` instead of `-C target-cpu=native`.
This enables all 40+ target features including AVX-512, VAES, VPCLMULQDQ,
GFNI, SHA.

### 28. pyzstd is now pure Python

**Non-issue**: pyzstd v0.19.1 restructured -- the C extension moved to a
separate `backports-zstd` package. The main `pyzstd` package is now pure
Python (`py3-none-any`). Since `zstandard` covers the same use case, pyzstd
was removed from the build list.

### 29. pyarrow requires full Arrow C++ build

**Non-issue**: pyarrow's source build requires the entire Apache Arrow C++
library pre-installed (30+ minute build with its own dependency tree). The
PyPI binary uses runtime SIMD dispatch and detects AVX-512 at startup, so
there's no meaningful gain from a source build.
