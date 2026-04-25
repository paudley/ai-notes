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

### 10b. libhipcxx atomic_codegen gated only on FileCheck

**Symptom**: TheRock configure fails in `math-libs/libhipcxx` with:
`Could not find cuobjdump using the following names: cuobjdump`

**Root cause**: libhipcxx's `test/atomic_codegen` cmake logic is guarded only
by whether `FileCheck` exists. On HIP-only ROCm builds, `/usr/bin/FileCheck`
is present, so cmake enters the CUDA-only atomic codegen test path and then
hard-requires NVIDIA's `cuobjdump`, aborting configuration.

**Fix**: Replace the `if (filecheck)` guard with
`if (filecheck AND LIBCUDACXX_ENABLE_CUDA)`, so those tests only configure
when the CUDA backend is actually enabled.

**Does this involve FileCheck?**: Yes. `FileCheck` is the trigger that causes
the CUDA-only atomic codegen directory to configure in the first place. The
fatal missing tool is still `cuobjdump`, but the bad guard is the
`if (filecheck)` check.

### 10c. roctx64 pre-hooks ignore `THEROCK_ENABLE_PROFILER=OFF`

**Symptom**: TheRock configure fails in RCCL (and potentially rocBLAS or
rocSPARSE) with:
`Could not find _therock_legacy_roctx64 using the following names: roctx64`

**Root cause**: Several TheRock pre-hook cmake files unconditionally run a
legacy `roctx64` link-patch on Linux. Our top-level configuration explicitly
sets `-DTHEROCK_ENABLE_PROFILER=OFF`, so the profiler stack and `roctx64`
library are intentionally absent. The pre-hooks should not hard-require
`roctx64` when profiling is disabled.

**Fix**: Gate the affected Linux pre-hooks on
`if(NOT WIN32 AND THEROCK_ENABLE_PROFILER)` instead of just `if(NOT WIN32)`.
This preserves the workaround when the profiler stack is enabled while
allowing profiler-disabled builds to configure normally.

### 10d. Disable ROCgdb in TheRock configure

**Symptom**: TheRock build later fails in `debug-tools/rocgdb` even though the
core ROCm compiler/runtime libraries continue building in parallel.

**Root cause**: ROCgdb is an optional ROCm debugger component. It is not needed
for building or running the vLLM/PyTorch/Triton stack, but TheRock enables it
by default as part of its debug-tools feature set.

**Fix**: Pass `-DTHEROCK_ENABLE_ROCGDB=OFF` during TheRock configure. This
skips the optional debugger subproject while preserving the rest of the ROCm
SDK required by the inference stack.

### 10e. RCCL enables ROCTX even when profiler is disabled

**Symptom**: RCCL configure later fails with:
`Could not find super-project find_library(NAMES roctx64)`

**Root cause**: RCCL's own cmake enables `ROCTX` by default. When `ROCTX` is
on, it calls `find_library(roctx64)` and `find_path(roctracer/roctx.h)`.
That conflicts with this build's `THEROCK_ENABLE_PROFILER=OFF` setting, where
the profiler stack and `roctx64` are intentionally unavailable.

**Fix**: Inject `-DROCTX=OFF` into TheRock's RCCL subproject cmake args so
RCCL skips its optional ROCTX tracing path entirely.

### 10f. RCCL rccl_common.h misses tuner macro definitions

**Symptom**: RCCL's `rccl_wrap.cc` fails during the TheRock build with:
`use of undeclared identifier 'NCCL_NUM_ALGORITHMS'` and
`use of undeclared identifier 'NCCL_NUM_PROTOCOLS'`

**Root cause**: Newer RCCL snapshots moved `NCCL_NUM_ALGORITHMS` and
`NCCL_NUM_PROTOCOLS` into `src/include/plugin/nccl_tuner.h`, but
`src/include/rccl_common.h` still references them after including only
`nccl_common.h`, `nccl.h`, `param.h`, and `core.h`. In the generated hipify
header tree that leaves the tuner macros undefined when `rccl_wrap.cc`
compiles.

**Fix**: Inject `#include "plugin/nccl_tuner.h"` into `rocm-systems/projects/rccl/src/include/rccl_common.h` so the
internal tuner macro definitions are available before the RCCL addon algorithm
and protocol declarations.

### 10g. RCCL nvtx.h ignores NVTX stub mode for direct includes

**Symptom**: RCCL's `group.cc` fails during the TheRock build with macro
redefinition warnings followed by `redefinition of 'nccl_domain'` between
`nvtx.h` and `nvtx_stub.h`.

**Root cause**: TheRock already passes `-DNVTX_NO_IMPL`, and RCCL's `core.h`
properly switches to `nvtx_stub.h` in that mode. However, some RCCL sources
include `nvtx.h` directly, and `nvtx.h` itself does not currently honor
`NVTX_NO_IMPL`, so both the real NVTX declarations and the stub declarations
end up active in the same translation unit.

**Fix**: Wrap `rocm-systems/projects/rccl/src/include/nvtx.h` itself in an
`#ifdef NVTX_NO_IMPL` guard that includes `nvtx_stub.h`, and extend
`rocm-systems/projects/rccl/src/include/nvtx_stub.h` with a no-op
`NCCL_NVTX3_FUNC_RANGE` definition so sources using the common NVTX macro
surface still compile in stub mode.

### 10h. hipBLASLt enables ROCTX markers by default

**Symptom**: hipBLASLt configure fails with:
`Could not find super-project find_library(NAMES roctx64)`

**Root cause**: hipBLASLt's cmake has `HIPBLASLT_ENABLE_MARKER` enabled by
default for shared-library builds. When enabled, it calls `find_library(roctx64)`
and hard-fails if roctracer is absent. That is incompatible with this build's
`THEROCK_ENABLE_PROFILER=OFF` configuration.

**Fix**: Inject `-DHIPBLASLT_ENABLE_MARKER=OFF` into TheRock's hipBLASLt
subproject cmake args so it skips the optional ROCTX marker path entirely.

### 10f2. hipSPARSELt enables ROCTX markers by default

**Symptom**: hipSPARSELt configure fails with:
`Could not find ROCTX64_LIBRARY using the following names: roctx64, libroctx64`

**Root cause**: hipSPARSELt's top-level cmake defines
`HIPSPARSELT_ENABLE_MARKER` as ON by default on non-Windows builds. When that
option is enabled, it unconditionally calls `find_library(ROCTX64_LIBRARY ...)`
and `find_path(roctracer/roctx.h ...)`. That conflicts with this build's
`THEROCK_ENABLE_PROFILER=OFF` configuration, where roctracer/roctx64 are
intentionally absent.

**Fix**: Inject `-DHIPSPARSELT_ENABLE_MARKER=OFF` into TheRock's hipSPARSELt
subproject cmake args so configure skips the optional ROCTX marker path.

### 10f3. MIOpen still probes roctracer headers when profiler is disabled

**Symptom**: MIOpen configure fails with:
`Could not find super-project find_path(NAMES roctracer/roctx.h)`

**Root cause**: MIOpen's top-level `CMakeLists.txt` sets
`MIOPEN_USE_ROCTRACER` to `ON` by default. On non-Windows builds that makes
configure call `find_path(roctracer/roctx.h)` and `find_library(roctx64)`,
which conflicts with this build's `THEROCK_ENABLE_PROFILER=OFF`
configuration where roctracer/roctx64 are intentionally absent.

**Fix**: Inject `-DMIOPEN_USE_ROCTRACER=OFF` into TheRock's MIOpen subproject
cmake args so the optional rocTracer/ROCTX probe is skipped entirely.

### 10g. rocprofiler-sdk yaml-cpp patch path moved in current TheRock layout

**Symptom**: rocprofiler-sdk later fails compiling vendored yaml-cpp with
undeclared `uint16_t` / `uint32_t` in `external/yaml-cpp/src/emitterutils.cpp`.

**Root cause**: The source patch for yaml-cpp's missing `<cstdint>` include was
targeting an old path layout and no longer matched the current TheRock source
tree. As a result, the workaround never applied.

**Fix**: Update the patch target path to
`rocm-systems/projects/rocprofiler-sdk/external/yaml-cpp/src/emitterutils.cpp`
so the existing `<cstdint>` insertion applies to the actual vendored file.

### 10h. rocBLAS still probes roctracer headers when profiler is disabled

**Symptom**: rocBLAS configure fails with:
`Could not find super-project find_path(... roctracer/roctx.h ...)`
even though TheRock already passes `-DROCTX=OFF` to the rocBLAS subproject.

**Root cause**: rocBLAS's `projects/rocblas/library/CMakeLists.txt` gates the
ROCTX probe only on `BUILD_SHARED_LIBS`. Its `find_path(roctracer/roctx.h)` and
`find_library(roctx64)` block does not honor the `ROCTX` cache option, so the
header/library probe still runs in shared-library builds even when profiling is
explicitly disabled.

**Fix**: Keep injecting `-DROCTX=OFF` from TheRock's `math-libs/BLAS/CMakeLists.txt`,
and patch `rocm-libraries/projects/rocblas/library/CMakeLists.txt` in two ways:
first, guard the ROCTX probe with `if(BUILD_SHARED_LIBS AND ROCTX)` instead of
only `if(BUILD_SHARED_LIBS)`; second, add `target_compile_definitions(... DISABLE_ROCTX)`
when `ROCTX` is off so sources like `logging.cpp` do not include
`<roctracer/roctx.h>` after the probe has been skipped.


### 10h2. rocSPARSE still probes roctracer headers unless BUILD_WITH_ROCTX is turned off

**Symptom**: rocSPARSE configure fails with:
`Could not find super-project find_path(... roctracer/roctx.h ...)`

**Root cause**: Unlike rocBLAS, rocSPARSE already has an upstream
`if(BUILD_SHARED_LIBS AND BUILD_WITH_ROCTX)` guard around its ROCTX probing in
`projects/rocsparse/library/CMakeLists.txt`. However, TheRock's
`math-libs/BLAS/CMakeLists.txt` does not pass `-DBUILD_WITH_ROCTX=OFF` to the
rocSPARSE subproject, so the guarded probe still stays enabled by default and
hits the explicit super-project finder when profiler components are absent.

**Fix**: Inject `-DBUILD_WITH_ROCTX=OFF` into TheRock's rocSPARSE subproject
cmake args so rocSPARSE skips the optional roctx header/library probe entirely.

### 10i. ROCR-Runtime OpenCL blit kernels do not know the device-lib path while bootstrapping

**Symptom**: TheRock build fails in `rocm-systems/projects/rocr-runtime` while
building `opencl_blit_objects` with a command like:
`clang -O2 -x cl ... imageblit_kernels.cl`
followed by:
`cannot find ROCm device library; provide its path via '--rocm-path' or '--rocm-device-lib-path'`

**Root cause**: `runtime/hsa-runtime/image/blit_src/CMakeLists.txt` hard-codes
a `clang -x cl` custom command for the OpenCL blit kernels, but it does not
pass any ROCm install root or device-library path. During a TheRock bootstrap
build, clang cannot infer the just-built `amdgcn/bitcode` directory on its own,
so current clang releases abort instead of compiling the embedded blit objects.

**Fix**: Patch `rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/image/blit_src/CMakeLists.txt`
to detect `${CMAKE_PREFIX_PATH}/llvm/amdgcn/bitcode` or
`${CMAKE_PREFIX_PATH}/amdgcn/bitcode` and append
`--rocm-device-lib-path=<detected path>` to the generated `clang -x cl` command.

### 25b. YAML patchelf RPATH expansion treated `$ORIGIN` as a shell variable

**Symptom**: `build-vllm.sh` aborts while applying `patchelf_rpath` patches
with:
`./build-vllm.sh: line 754: ORIGIN: unbound variable`

**Root cause**: The YAML-driven patch helper expanded RPATH templates with
`eval echo`. That works for `${LOCAL_PREFIX}` but also tries to expand ELF
loader tokens like `$ORIGIN` as shell variables, which fails under
`set -u`.

**Fix**: Escape `$ORIGIN` before running `eval echo` so shell variable
expansion still resolves our repo variables while preserving the literal ELF
RPATH token for `patchelf`.

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

**Root cause**: PyTorch's ROCm build injects `-fclang-abi-compat=17` in two
places: top-level host flags in `CMakeLists.txt` and HIP flags in
`cmake/Dependencies.cmake`. On current amdclang releases that Clang 17 ABI
override can still leave host/HIP objects mangled inconsistently, producing
undefined symbols between `libtorch_cpu.so` and `libtorch_hip.so`.

**Fix**: Remove both `-fclang-abi-compat=17` injections — the host-side
`append_cxx_flag_if_supported(... CMAKE_CXX_FLAGS)` line in `CMakeLists.txt`
and the HIPCC line in `cmake/Dependencies.cmake`.

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

## PyTorch Wheel Fixes (Phase C, Step 10)

### 24. numpy>=2 ABI compatibility

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

### 25. PyTorch .so patches not baked into wheel

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

### 25b. Build tree RPATH leak in libtorch_python.so

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

### 25c. Validation retry for stale installed torch artifacts

**Symptom**: Step 12 (`Validate PyTorch GPU access`) fails on `import torch`
with `libtorch_hip.so: undefined symbol: _ZN2at4cuda4blas4gemm...` even
though the wheel was rebuilt from a clean `build/` directory.

**Root cause**: In some failed/retried builds, the virtualenv can still hold
stale `torch/`, `torch-*.dist-info`, or `functorch/` artifacts from an older
install. When Python resolves mixed old/new extension modules from the same
environment, `libtorch_hip.so` can be loaded without the matching symbol
provider from the rebuilt wheel.

**Fix**: During validation, detect this exact unresolved-symbol import
failure, remove old torch artifacts from the active site-packages directory,
reinstall the newest locally built torch wheel with `uv pip install
--force-reinstall --no-deps`, and retry the import once before failing.

### 25d. NumPy 2.0 ABI target version

**Symptom**: `import torch` crashes with "A module that was compiled using
NumPy 1.x cannot be run in NumPy 2.4.3".

**Root cause**: PyTorch's `numpy_stub.h` does NOT set `NPY_TARGET_VERSION`.
Without this, numpy 2.x headers compile against the oldest compatible API
(1.20 / `0x0e`) by default, producing a `.so` that fails the runtime ABI
check when loaded with numpy >= 2.0.

**Fix**: Patch `torch/csrc/utils/numpy_stub.h` to set `NPY_TARGET_VERSION`
before including `<numpy/arrayobject.h>`:

### 25e. Missing OpenMP runtime linkage in libtorch_cpu.so

**Symptom**: `import torch` fails immediately with:

```text
ImportError: .../torch/lib/libtorch_cpu.so: undefined symbol: __kmpc_fork_call
```

**Root cause**: PyTorch was built with clang/amdclang OpenMP enabled, so
`libtorch_cpu.so` references the `__kmpc_*` entry points provided by LLVM's
OpenMP runtime (`libomp.so` / `libiomp5.so`). In some builds the packaged wheel
omits that runtime from `libtorch_cpu.so`'s `NEEDED` list, so the dynamic
linker never loads it during `import torch`.

**Fix**: Do **not** rewrite PyTorch's internal ELF dependency graph to force in
an OpenMP runtime. That proved brittle and could trigger follow-on import
failures in `libtorch_hip.so`. Instead, preload LLVM's OpenMP runtime from the
environment whenever it exists under `${LOCAL_PREFIX}/lib/llvm/lib` or
`${LOCAL_PREFIX}/lib`:

```bash
export LD_PRELOAD=/opt/src/vllm/local/lib/llvm/lib/libomp.so${LD_PRELOAD:+:$LD_PRELOAD}
```

```c
#ifndef NPY_TARGET_VERSION
#define NPY_TARGET_VERSION 0x00000012  /* NPY_2_0_API_VERSION */
#endif
```

The hex value must be used directly because `NPY_2_0_API_VERSION` is defined
inside `numpyconfig.h` which is included by `arrayobject.h`.

### 25d. PyTorch wheel misses the LLVM OpenMP runtime path/dependency

**Symptom**: `import torch` fails immediately with:
`ImportError: .../libtorch_cpu.so: undefined symbol: __kmpc_fork_call`

**Root cause**: The source-built PyTorch wheel is compiled with amdclang's
OpenMP runtime, but the packaged wheel patching only added
`${LOCAL_PREFIX}/lib` to RPATH and only injected `librocm_smi64.so` into
`libtorch_hip.so`. The OpenMP runtime lives under
`${LOCAL_PREFIX}/lib/llvm/lib`, and some wheel builds also leave
`libtorch_cpu.so` without an explicit `libomp.so` NEEDED entry. That leaves
the `__kmpc_*` symbols unresolved at import time.

**Fix**: During the unpack/patch/repack step, rewrite PyTorch wheel RPATHs to
include both `${LOCAL_PREFIX}/lib` and `${LOCAL_PREFIX}/lib/llvm/lib`, and add
`libomp.so` to `libtorch_cpu.so` when it still exports unresolved
`__kmpc_fork_call` without an existing `libomp` dependency.

## TorchVision (Phase C, Steps 12-13)

### 26. TorchVision source build

**Non-issue**: TorchVision is now built from source (steps 12-13) against
the source-built PyTorch to ensure ABI compatibility. Uses amdclang from
TheRock. CPU-only (no CUDA/ROCm GPU ops -- TorchVision's GPU kernels are
not needed for inference).

## Flash Attention (Phase F, Steps 26-28)

### 27. amdsmi import order (flash_attn)

**Symptom**: Same as vLLM (#35 Patch 1) — amdsmi C extension crash when
loaded after torch.

**Root cause**: Identical to the vLLM fix. flash_attn's `__init__.py`
imports torch before amdsmi, causing the same C extension initialization
conflict.

**Fix**: Prepend `import amdsmi` before any torch imports in
`flash_attn/__init__.py`.

## Wheel Builds (Phase H)

### 28. cmake pip wrapper in build isolation

**Symptom**: sentencepiece source build fails with `ImportError: No module
named 'cmake'`.

**Root cause**: The `cmake` pip package installs a Python wrapper at
`.venv/bin/cmake` that does `from cmake import cmake`. Inside pip's build
isolation, the cmake Python module isn't available, so the wrapper fails.

**Fix**: Replace the Python wrapper with a symlink to the real system cmake
binary (`/usr/bin/cmake`).

### 29. meson -Werror vs -mllvm flags (numpy build)

**Symptom**: numpy build fails on meson capability probes.

**Root cause**: meson hard-codes `-Werror=unused-command-line-argument` in
`ClangCompiler.get_compiler_check_args()` AFTER our CFLAGS. Driver-level
`-mllvm` flags are reported as "unused" in compile-only checks (`-c`),
killing every meson capability probe.

**Fix**: Transform `-mllvm <arg>` to `-Xclang -mllvm -Xclang <arg>` in
CFLAGS for wheel builds. `-Xclang` passes flags directly to the compiler
frontend/backend, bypassing the driver's argument tracking. Move `-famd-opt`
to LDFLAGS (link-time only, no-op at compile time).

### 30. Rust + amdclang linker

**Symptom**: Cargo fails to link with "cc: error: unrecognized argument".

**Root cause**: AMD's `cc` symlink (created in step 4) rejects binaries
not prefixed by "amd". Cargo uses `cc` by default.

**Fix**: Set `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=amdclang`.
Also unset CFLAGS/CXXFLAGS/LDFLAGS because they contain clang-specific
flags that Rust's internal `cc` invocations don't understand.

### 31. Rust -C target-cpu=native bug

**Symptom**: Rust binary only uses SSE2 despite running on Zen 5.

**Root cause**: Rust's native CPU detection identifies znver5 but only
enables SSE2 features (a known `rustc` bug).

**Fix**: Use explicit `-C target-cpu=znver5` instead of `-C target-cpu=native`.
This enables all 40+ target features including AVX-512, VAES, VPCLMULQDQ,
GFNI, SHA.

### 32. pyzstd is now pure Python

**Non-issue**: pyzstd v0.19.1 restructured -- the C extension moved to a
separate `backports-zstd` package. The main `pyzstd` package is now pure
Python (`py3-none-any`). Since `zstandard` covers the same use case, pyzstd
was removed from the build list.

### 33. pyarrow requires full Arrow C++ build

**Non-issue**: pyarrow's source build requires the entire Apache Arrow C++
library pre-installed (30+ minute build with its own dependency tree). The
PyPI binary uses runtime SIMD dispatch and detects AVX-512 at startup, so
there's no meaningful gain from a source build.

## vLLM Runtime Patches (Phase E, Step 20b)

These patches fix runtime issues specific to gfx1151 (RDNA 3.5, wave32).
They are applied after cloning vLLM and before building. "Patch N" numbers
in parentheses refer to the YAML `packages.vllm.patches[]` index.

### 34. amdsmi import order (Patch 1)

**Symptom**: `segfault` or `ImportError` on `import torch` when amdsmi is
installed.

**Root cause**: amdsmi's C extension conflicts with torch's ROCm
initialization if loaded after torch. Both bind to the same ROCm SMI
shared library but expect different initialization states.

**Fix**: Prepend `import amdsmi` before any torch imports in vLLM's
`__init__.py`. Identical to the flash_attn fix (#27).

### 35. AITER gate extension to gfx1x (Patches 2-5)

**Symptom**: AITER optimizations (attention, GEMM, normalization) are
silently disabled on gfx1151 — only eager PyTorch paths are used.

**Root cause**: vLLM upstream gates AITER behind `on_gfx9()` checks
(MI300X architecture family). gfx1151 is RDNA 3.5, not gfx9.

**Fix**: Extend `is_aiter_found_and_supported()` in `_aiter_ops.py` and
`supports_compute_capability()` in `rocm_aiter_fa.py` to accept
`on_gfx1x()` alongside `on_gfx9()`. AITER has explicit gfx1151 tuning
(chip_info.py enum 13, BLOCK_M/N=32, waves_per_eu=2).

### 36. ViT attention revert to gfx9-only (Patch 6)

**Symptom**: Vision Transformer (ViT) encoder attention crashes with
"invalid argument for fmha_fwd" on gfx1151.

**Root cause**: The CK fmha_fwd kernel rejects ViT-specific attention
dimensions (head_dim/seq_len combinations) on gfx1151. The decoder
attention path (unified + FA) works correctly on gfx1x, but the ViT
encoder path cannot use CK attention.

**Fix**: Keep ViT attention gated to `on_gfx9()` only. On gfx1151, ViT
falls through to `TRITON_ATTN` which works correctly. If a previous build
had extended the gate, this patch reverts it.

### 37. FP8 linear disable on gfx1x (Patch 7)

**Symptom**: GPU page fault crash during FP8 quantized inference.

**Root cause**: CK GEMM kernels (`module_gemm_a8w8_blockscale`) compile
for gfx1151 but use CDNA-specific MFMA (Matrix Fused Multiply-Add)
instructions that don't exist on RDNA 3.5. The kernel executes illegal
instructions, causing page faults.

**Fix**: Add gfx1x guard to `is_linear_fp8_enabled()` in `_aiter_ops.py`.
Returns `False` on RDNA 3.x, forcing vLLM to use its Triton blockscale
GEMM fallback which generates correct gfx1151 kernels.

### 38. AttrsDescriptor `__repr__` for Inductor codegen (Triton Patch 2, vLLM inline)

**Symptom**: `SyntaxError` when loading torch.compile-generated Triton
kernel files. torch.compile works on first run but fails on cache reload.

**Root cause**: torch Inductor's codegen uses `{triton_meta!r}` to
serialize kernel metadata into generated Python source. The ROCm Triton
fork's `AttrsDescriptor` class has no `__repr__`, so Python falls back to
`object.__repr__()` producing `<triton.backends.compiler.AttrsDescriptor
object at 0x...>` — invalid Python syntax that causes `SyntaxError` when
the generated file is re-imported.

**Fix**: Add `__repr__` to `AttrsDescriptor` in
`triton/backends/compiler.py` that produces valid, round-trippable Python
via `from_dict()`. Applied in two places: (1) Triton source tree during
`build_triton()` (YAML triton patch 2), and (2) the installed triton
package during `patch_vllm_gfx1151()` to catch pre-built wheels. With this
patch, `torch.compile` works correctly on gfx1151 — `--enforce-eager` is
NOT required.

### 39. Duplicate pattern registration crash (Patch 9)

**Symptom**: `RuntimeError: Duplicate pattern` during torch.compile
initialization with AITER fusion passes enabled.

**Root cause**: `RocmAiterRMSNormQuantFusionPass` in
`rocm_aiter_fusion.py` registers patterns in a loop over
`epsilon x match_aiter_quant` combinations. Some combinations produce
identical pattern graphs, and `torch._inductor.pattern_matcher` raises
on duplicates.

**Fix**: Add `skip_duplicates=True` to all `pm.register_replacement()`
calls in the fusion pass.

### 40. `+rms_norm` custom_ops block on gfx1x (Patch 8)

**Symptom**: Model produces garbage/incoherent output with AITER enabled
and torch.compile active. Correct output in eager mode.

**Root cause**: When AITER RMSNorm is detected, vLLM's `rocm.py` adds
`+rms_norm` to the `custom_ops` list. This tells torch.compile/Inductor
to treat RMSNorm as an **opaque barrier** in the compute graph rather
than an inline operation. On gfx1x (RDNA 3/4, wave32), Inductor generates
incorrect code at the graph partition boundaries created by this barrier.

Both the CK and Triton RMSNorm kernels are correct in isolation — the bug
is purely in how Inductor restructures the compute graph when RMSNorm is
declared as an opaque custom op. The effect is subtle: the model runs
without errors but produces nonsensical output.

**Fix**: Add `and not on_gfx1x()` guard to the `+rms_norm` insertion in
`rocm.py`. RMSNorm stays inline in the Inductor graph and gets fused
normally by the compiler. This was the single most impactful fix:

| Model | Before (PIECEWISE, no AITER) | After (FULL graph, ALL AITER) | Speedup |
|-------|------------------------------|-------------------------------|---------|
| Qwen2.5-0.5B | 137.4 tok/s | 1059.8 tok/s | 7.7x |
| Qwen2.5-1.5B | 44.2 tok/s | 391.6 tok/s | 8.9x |

The speedup comes from enabling FULL CUDA graph capture (entire forward
pass as a single HIPGraph) combined with ALL AITER optimizations
(attention, GEMM, normalization). Previously, the `+rms_norm` bug forced
PIECEWISE graph mode with AITER disabled.

### 41. Triton sampler page fault on gfx1151 (historical, build-specific)

**Symptom**: Some builds observed GPU page faults during top-k/top-p sampling
after torch.compile AOT compilation on RDNA 3.5.

**Root cause**: This is not a universal hardware limitation. It is a stack
compatibility issue (specific vLLM/PyTorch/Triton/AITER combinations and
cached compile artifacts), where the compiled sampler path can diverge from
its eager behavior. On other builds of the same hardware/OS, the Triton
sampler works correctly.

**Current approach**: Do **not** hard-disable Triton sampler in the build
patch set. Keep upstream behavior by default and treat sampler faults as an
investigation target (version skew, patch skew, stale compile cache, and
backend selection), not a blanket architecture rule.

### 42. FLA chunk_delta_h autotuner + exp() type inference (Patches 11-15)

**Symptom**: Two issues in FLA (Flash Linear Attention) Triton kernels:
1. Page faults during autotuning with `num_stages>2` or `BV=64`
2. Invalid IR from `exp()` type inference on HIP

**Root cause**: The chunk_delta_h Triton kernel's autotuner tries pipeline
depths (stages=2,3,4) and block sizes (BV=32,64) that exceed RDNA 3.5's
register pressure limits, causing page faults. Separately, the HIP Triton
compiler fails to infer types for `exp(scalar_bf16 - block_ptr_load_bf16)`,
generating invalid intermediate representation.

**Fix**:
- Restrict AMD autotuning to `num_stages=2` and `BV=32` only (via
  `is_amd` flag)
- Cast `exp()` operands to `tl.float32` explicitly, which also improves
  precision

### 43. Qwen3.5 FLA warmup page fault for T < BT (Patch 16)

**Symptom**: Page fault during Qwen3.5-next model warmup when FLA kernels
are called with sequence lengths T=16 or T=32.

**Root cause**: On RDNA 3.5 (wave32), `tl.make_block_ptr` page-faults when
the sequence length T is less than the chunk size BT (64). HIP materializes
the out-of-bounds address computation that CDNA (wave64) handles
differently. The warmup loop iterates `T in (16, 32, 64)` but only T=64
(where T==BT) is safe.

**Fix**: Restrict the warmup loop in `qwen3_next.py` to `for T in (64,)`
only.

### 44. flash_attn_2_cuda import on ROCm (Patch 17)

**Symptom**: `ModuleNotFoundError: flash_attn_2_cuda` when loading rotary
embedding with flash_attn installed.

**Root cause**: On ROCm, flash_attn is a pure-Python wheel that provides
Triton-based kernels via `flash_attn.ops.triton.*` but does NOT include the
CUDA native extension `flash_attn_2_cuda`. The import chain
`flash_attn.ops.triton.rotary` -> `flash_attn_2_cuda` fails because the
`.so` doesn't exist.

**Fix**: Wrap the import in `rotary_embedding/common.py` with a try/except
for `ImportError`/`ModuleNotFoundError`. When the native extension is
absent, the Triton-based rotary path is still available through other code
paths.

### 45. AITER RMSNorm CK dispatch on gfx1x (Patches 18-19)

**Symptom**: Illegal instruction crash during quantized inference when AITER
RMSNorm is active on gfx1151.

**Root cause**: The CK (Composable Kernel) RMSNorm implementations
(`rocm_aiter.rmsnorm2d_fwd_with_dynamicquant` and
`rmsnorm2d_fwd_with_add_dynamicquant`) use CDNA-specific assembly (MFMA
instructions) that doesn't exist on RDNA 3.5. Additionally, the CK versions
accept a `use_model_sensitive_rmsnorm=0` kwarg that the Triton versions
don't.

**Fix**: Add architecture dispatch in `_aiter_ops.py`. On `on_gfx1x()`, use
the Triton RMSNorm from `aiter.ops.triton.normalization.rmsnorm` (which
generates correct wave32 kernels). On gfx9 (CDNA), use the original CK path
with the `use_model_sensitive_rmsnorm=0` kwarg.

### 52. Flash Attention internal AITER install failure (Step 28)

**Symptom**: Flash Attention build fails with `error: [Errno 2] No such file
or directory: 'aiter_meta/hsa/gfx942/fmoe_2stages/...'`.

**Root cause**: Flash Attention's `setup.py` (ROCm `main_perf` branch) runs
`subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation",
"third_party/aiter"])` during the build. This AITER submodule bundles
pre-compiled `.co` (code object) files for gfx942 only. On gfx1151 the gfx942
code objects are missing from the source tree, causing `FileNotFoundError`
during `setup.py`'s `package_data` collection.

**Fix**: Patch `setup.py` to replace the `subprocess.run(pip install aiter)`
call with `pass`. We build AITER separately from the PyTorch submodule source
(step 28b) with proper gfx1151 patches, so the flash_attn internal install
is unnecessary and harmful.

### 53. AITER JIT pre-warm SystemExit propagation (Step 29b)

**Symptom**: Build script exits non-zero during AITER JIT pre-warm when any
module fails to compile, killing subsequent build steps.

**Root cause**: AITER's `build_module()` function (`aiter/jit/core.py:694`)
raises `SystemExit` on compilation failure. `SystemExit` inherits from
`BaseException`, not `Exception`, so the warmup script's `except Exception`
clause doesn't catch it. The `SystemExit` propagates out of the Python
interpreter, making it exit non-zero.

**Fix**: Change the warmup script's exception handler from `except Exception`
to `except (Exception, SystemExit)`. Expected failures (CDNA-only modules like
`module_activation`, `module_quick_all_reduce`, `module_moe_asm`) are caught,
counted, and logged without terminating the build.

### 54. Optimized wheels not installed into venv (Steps 30-31)

**Symptom**: After build completes, the venv has pip-resolved versions of
numpy (2.1.3), and lacks orjson and asyncpg entirely, despite Zen 5-optimized
wheels being present in `wheels/`.

**Root cause**: Steps 30-31 build optimized wheels into `wheels/` but never
install them back into the build venv. The wheels were only intended for
distribution to other environments.

**Fix**: Add `uv pip install --force-reinstall --no-deps <wheel>` after each
wheel is built (or confirmed to exist). This replaces the venv's
pip-installed versions with the source-built, Zen 5-optimized versions.

### 55. Triton stdout pollution breaks AITER JIT directory capture (Step 29b)

**Error**: After AITER JIT pre-warm completes, the build script dies with exit
code 1 instead of continuing to steps 30-35.

**Root cause**: `get_user_jit_dir()` imports triton, which prints a warning to
**stdout** (not stderr): `Warning: triton.experimental.gluon or
triton.experimental.gluon.language not exists...`. The bash `$(...)` captures
this warning along with the actual JIT directory path, creating a multi-line
`jit_dir` variable. When `find "${jit_dir}" -maxdepth 1 ...` runs, the garbage
path causes `find` to fail, and `set -euo pipefail` kills the script.

**Fix**: Pipe the `get_user_jit_dir()` output through `tail -1` to grab only
the last line (the actual path), discarding any stdout pollution from upstream
imports. Applied to all three `jit_dir` capture sites (lines 2372, 3037, 3204).

## AITER Source Rebuild (Phase F, Step 28b)

### 46. AITER CK ABI mismatch

**Symptom**: JIT compilation of AITER MHA kernels fails with ABI
mismatches -- struct field types, missing members, narrowing conversion
errors.

**Root cause**: AITER's MHA (Multi-Head Attention) kernels use CK
(Composable Kernel) tile headers for JIT compilation at runtime. The
pip-installed aiter wheel includes pre-compiled `.cu` interface files built
against a specific CK commit. If `CK_DIR` points to a different CK version,
the compiled interfaces and runtime JIT headers disagree on struct layouts,
causing compilation failures.

**Fix**: Step 28b rebuilds AITER from the PyTorch submodule source tree
(`pytorch/third_party/aiter`) with `CK_DIR` pointing to the matching CK
submodule. This ensures the compiled `.cu` interfaces and CK headers are
from the same commit. The stale JIT cache is cleared before rebuild.

### 47. AITER vec_convert.h CDNA-only packed ISA (gfx1151 header patch)

**Symptom**: JIT compilation fails with "invalid instruction" for AITER
kernels that use packed FP8 conversion on gfx1151.

**Root cause**: `ck_tile/vec_convert.h` contains three CDNA-only packed
instructions:
- `v_pk_mul_f32` (packed FP32 multiply, gfx940+ only)
- `v_cvt_pk_fp8_f32` (packed FP8 convert, gfx942+ only)
- `v_cvt_pk_bf8_f32` (packed BF8 convert, gfx942+ only)

These are inline assembly instructions that RDNA 3/3.5 hardware cannot
execute.

**Fix**: Replace with architecture-dispatched code using
`CK_TILE_RDNA3_NO_PK_FP8` preprocessor guard. On RDNA 3/3.5
(`__gfx11xx__`), scalar C++ equivalents are used instead of packed assembly.

### 48. AITER hip_reduce.h DPP broadcast instructions (gfx1151 header patch)

**Symptom**: Illegal instruction during warp reduction operations in AITER
kernels on gfx1151.

**Root cause**: `hip_reduce.h` uses two DPP (Data Parallel Primitives)
broadcast instructions:
- `row_bcast:15` (0x142) -- cross-row broadcast, CDNA only
- `row_bcast:31` (0x143) -- cross-half broadcast, CDNA only

These DPP modes don't exist on RDNA, which uses a different warp shuffle
mechanism.

**Fix**: Replace with `ds_swizzle` (`warp_swizzle<T, 0x1e0>`) matching
rocprim's own `warp_reduce_dpp.hpp` RDNA path. The `WarpSize > 32` path
uses a `static_assert` since RDNA is wave32-only (CDNA is wave64). Patches
target installed site-packages headers (not source tree) because AITER's
JIT reads from the venv.

### 49. FLA chunk_o autotuner page fault on AMD HIP (Patches 20-23)

**Symptom**: GPU page fault during Triton autotuning of the
`chunk_fwd_kernel_o` kernel in the FLA (Flash Linear Attention) ops for
Qwen3.5 GDN (Gated Delta Network) layers.

**Root cause**: Same class of issue as chunk_delta_h (#37). The autotune
search space includes BK/BV=64/128 and pipeline depths num_stages=3,4 that
exceed RDNA 3.5's register pressure limits. The kernel page-faults during
autotuning with the larger tile configurations.

**Fix**: Four sed patches to `chunk_o.py`:
1. Add `is_amd` to the utils import
2. Restrict BK to `[32]` on AMD (vs `BKV_LIST = [64, 128]`)
3. Restrict BV to `[32]` on AMD
4. Restrict num_stages to `[2]` on AMD (vs `[2, 3, 4]`)

### 50. KV cache page size mismatch: ROCm block_size vs hybrid alignment (Patches 24-25)

**Symptom**: Qwen3.5 GDN (hybrid mamba+attention model) fails with
assertion errors or incorrect generation due to block_size mismatch between
AITER's requirement and the hybrid model's mamba state alignment.

**Root cause**: The vLLM configuration pipeline has a sequencing issue:
1. `HybridAttentionMambaModelConfig.verify_and_update_config()` computes
   `attn_block_size` as lcm(mamba_state, kernel_alignment=32), producing
   e.g. 576 for Qwen3.5
2. `current_platform.check_and_update_config()` runs AFTER and sets
   `block_size=64` (ROCm AITER's requirement), clobbering step 1
3. The mamba layers now get a block_size (64) that doesn't satisfy their
   state alignment requirement

**Fix**: Two patches:
- **Patch 24** (`config/vllm.py`): Re-run
  `HybridAttentionMambaModelConfig.verify_and_update_config()` after the
  platform config, so the hybrid alignment is recomputed with the
  platform's block_size as a constraint
- **Patch 25** (`models/config.py`): Use
  `max(kernel_block_alignment_size, cache_config.block_size)` as the kernel
  alignment. If the platform already set block_size=64, the computed
  attn_block_size will be a multiple of both 64 (AITER) and the mamba state
  size

### 51. AITER unified attention Triton kernel crash on non-power-of-2 block_size (Patches 26-28) [TESTING]

**Status**: TESTING — routing hybrid models away from AITER attention
entirely (Patch 26) may be too aggressive. An alternative approach would be
to fix the AITER kernel to decouple TILE_SIZE from block_size, similar to
how `TritonAttentionBackend` handles it. Patches 27-28 are
defense-in-depth and may be sufficient on their own.

**Symptom**: `OutOfResources: shared memory, Required: 1081344, Hardware
limit: 65536` crash when AITER unified attention is used with hybrid models
that produce non-power-of-2 block_size (e.g. 576).

**Root cause**: AITER's unified attention Triton kernel uses
`TILE_SIZE = block_size` directly in `tl.arange()`, which requires N to be
a power of 2. After fix #50, block_size=576 (not power of 2).
`next_power_of_2(576) = 1024`, and the resulting shared memory allocation
(1024 * head_size * elem_size per K/V tile) exceeds the 64 KiB LDS on all
AMD GPUs.

**Fix**: Three-layer defense-in-depth:
- **Patch 26** (`rocm.py`) [TESTING]: Detect hybrid models via
  `model_config.is_hybrid` and skip AITER unified attention and AITER FA
  backends entirely. Hybrid models fall through to `TRITON_ATTN`, which
  decouples tile size from block size
- **Patch 27** (`rocm_aiter_unified_attn.py`) [TESTING]: Add power-of-2
  constraint to `supports_block_size()`. The original check only validated
  `block_size % 16 == 0`; now also requires
  `(block_size & (block_size - 1)) == 0`
- **Patch 28** (AITER `unified_attention.py`) [TESTING]: Cap
  `TILE_SIZE = min(block_size, 128)` in both `select_2d_config` and
  `select_3d_config`. For standard block sizes (64/128) this is a no-op.
  For abnormal block sizes that somehow reach the kernel, the cap prevents
  the LDS overflow

## Runtime Environment Files (Phase I)

The build generates `.env` files for llama.cpp backends used by Lemonade.
These are generated from `vllm-packages.yaml` via the `generate_env_file()`
helper — the YAML `packages.llamacpp.backends.{rocm,vulkan}.env` maps are
the single source of truth.

### ROCm backend `.env`

| Variable | Value | Purpose |
|----------|-------|---------|
| `HSA_OVERRIDE_GFX_VERSION` | `11.5.1` | Override for ROCm runtime gfx1151 detection |
| `ROCBLAS_USE_HIPBLASLT` | `1` | Use hipBLASLt for GEMM (faster on gfx1151) |
| `THP` | `always` | Transparent Huge Pages for unified memory |
| `LLAMA_ARG_BATCH` | `2048` | +33% prefill throughput over default (512) |
| `LLAMA_ARG_UBATCH` | `2048` | Micro-batch size matching batch size |

**Note**: Q8 KV cache (`LLAMA_ARG_CACHE_TYPE_K/V=q8_0`) is omitted from
the generated `.env`. It halves KV bandwidth on unified memory but causes
context creation failures on some small models (e.g. Qwen2.5 0.5B FP16).
Enable per-model during benchmarks.

### Vulkan backend `.env`

| Variable | Value | Purpose |
|----------|-------|---------|
| `LLAMA_ARG_BATCH` | `2048` | Batch size optimization |
| `LLAMA_ARG_UBATCH` | `2048` | Micro-batch size matching batch size |

No HSA/ROCm variables needed — Vulkan uses its own driver stack.
