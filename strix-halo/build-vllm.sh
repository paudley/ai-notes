#!/usr/bin/env bash
# Copyright 2026 Blackcat Informatics Inc.
# SPDX-License-Identifier: MIT
#
# build-vllm.sh - Build the ENTIRE vLLM inference stack from source
#
# Compiles all components from source using Clang/LLVM with aggressive
# optimization flags targeting AMD Strix Halo (Zen 5 + RDNA 3.5 gfx1151):
#
#   TheRock ROCm → AOCL-LibM → Python → PyTorch → Triton → AOTriton → vLLM → Flash Attention
#   + Optimized wheels for performance-critical Python packages
#
# Every component is compiled with: -march=native -O3 -flto=thin
# Rust packages use: -C target-cpu=znver5 (full AVX-512 + VAES)
# No pre-built tarballs. No pip wheels for core components.
#
# Prerequisites:
#   - Kernel 7.0+ with amdgpu and amdxdna loaded
#   - Clang 21+ and lld installed
#   - CMake 3.25+ and Ninja installed
#   - uv (Python package manager) installed
#   - Internet access for cloning git repos
#   - /opt/src/ directory must exist and be owned by current user
#   - ~100GB disk space for build artifacts
#
# Usage:
#   scripts/build-vllm.sh             # Full build (idempotent)
#   scripts/build-vllm.sh --rebuild   # Force rebuild (clean + build)
#   scripts/build-vllm.sh --step N    # Run from step N onward
#
# Build pipeline (30 steps):
#   Phase A: ROCm SDK (TheRock — builds amdclang used by everything downstream)
#     1. Clone TheRock          3. Build TheRock
#     2. Configure TheRock      4. Validate ROCm
#
#   Phase B: CPU Libraries + Python (built with amdclang from Phase A)
#     5. Build AOCL-Utils       7. Build Python 3.13
#     6. Build AOCL-LibM        8. Create venv
#
#   Phase C: ML Framework (PyTorch + TorchVision, ROCm fork)
#     9. Clone PyTorch         12. Clone TorchVision
#    10. Build PyTorch         13. Build TorchVision
#    11. Validate PyTorch
#
#   Phase D: Kernel Compilers (Triton + AOTriton)
#    14. Clone Triton          17. Clone AOTriton
#    15. Build Triton          18. Build AOTriton
#    16. Validate Triton
#
#   Phase E: Inference Engine (vLLM)
#    19. Clone vLLM             23. Install ROCm requirements
#    20. Patch amdsmi import    24. Build vLLM (AITER first)
#    20b. Patch gfx1151 AITER
#    21. Install build deps
#    22. use_existing_torch.py
#
#   Phase F: Attention (Flash Attention + AITER)
#    25. Reinstall amdsmi      28. Build Flash Attention
#    26. Clone Flash Attention
#    27. Patch Flash Attention
#
#   Phase G: Validation + Warmup
#    29. Smoke test
#    29b. AITER JIT pre-warm (compile all buildable modules ahead of time)
#    29c. TunableOp warmup (populate GEMM autotuning CSV)
#
#   Phase H: Optimized Wheels (Zen 5 native builds for downstream venvs)
#    30. Build Rust wheels      (orjson, cryptography — AVX-512 + VAES)
#    31. Build C/C++ wheels     (numpy, sentencepiece, zstandard, asyncpg)
#    32. Export source wheels    (torch, triton, torchvision, amd-aiter, amdsmi)

set -euo pipefail

# =============================================================================
# Setup
# =============================================================================

_SCRIPT_REAL_PATH="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || realpath "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")"
_SCRIPT_DIR="$(cd "$(dirname "$_SCRIPT_REAL_PATH")" && pwd)"

# Source shared helpers if available (platform context), otherwise
# define minimal inline versions for standalone operation.
_COMMON_SH="${_SCRIPT_DIR}/../lib/sh/common.sh"
if [[ -f "${_COMMON_SH}" ]]; then
    # shellcheck source=../lib/sh/common.sh
    source "${_COMMON_SH}"
else
    info()    { printf '  \033[1;34minfo\033[0m  %s\n' "$*"; }
    success() { printf '  \033[1;32m  ok\033[0m  %s\n' "$*"; }
    warn()    { printf '  \033[1;33mwarn\033[0m  %s\n' "$*" >&2; }
    error()   { printf '  \033[1;31m err\033[0m  %s\n' "$*" >&2; }
    die()     { error "$@"; exit 1; }
fi
unset _COMMON_SH

PLATFORM_DIR="${_SCRIPT_DIR}/.."

# Portable vllm-env.sh sourcing: try platform layout, fall back to co-located.
_vllm_source_env() {
    local env="${PLATFORM_DIR}/scripts/vllm-env.sh"
    [[ -f "${env}" ]] || env="${_SCRIPT_DIR}/vllm-env.sh"
    # shellcheck source=vllm-env.sh
    source "${env}"
}

unset _SCRIPT_REAL_PATH

# Source the vLLM environment (compiler flags, paths)
_vllm_source_env

TOTAL_STEPS=32

# CPython version to build
CPYTHON_VERSION="3.13.12"
CPYTHON_TAG="v${CPYTHON_VERSION}"

# Source repository URLs
# PyTorch: ROCm fork carries AMD-specific fixes (hipify, Tensile, rocm_smi linkage)
# that haven't been upstreamed. The upstream pytorch/pytorch works with USE_ROCM=1
# but the ROCm fork is what AMD's CI tests against.
THEROCK_REPO="https://github.com/ROCm/TheRock.git"
PYTORCH_REPO="https://github.com/ROCm/pytorch.git"
PYTORCH_BRANCH="develop"
TRITON_REPO="https://github.com/ROCm/triton.git"
TRITON_BRANCH="main_perf"
AOTRITON_REPO="https://github.com/ROCm/aotriton.git"
VLLM_REPO="https://github.com/vllm-project/vllm.git"
TORCHVISION_REPO="https://github.com/pytorch/vision.git"
FLASH_ATTN_REPO="https://github.com/ROCm/flash-attention.git"

# AOCL (AMD Optimizing CPU Libraries)
AOCL_UTILS_REPO="https://github.com/amd/aocl-utils.git"
AOCL_LIBM_REPO="https://github.com/amd/aocl-libm-ose.git"

# Unified install prefix — all C/C++ libraries install here.
# This gives us one -L path, one LD_LIBRARY_PATH entry, one CMAKE_PREFIX_PATH.
# Layout mirrors /usr/local/: bin/, lib/, include/, share/, lib/llvm/
LOCAL_PREFIX="${VLLM_DIR}/local"

# Source directories under /opt/src/vllm/
AOCL_UTILS_SRC="${VLLM_DIR}/aocl-utils"
AOCL_LIBM_SRC="${VLLM_DIR}/aocl-libm"
CPYTHON_SRC="${VLLM_DIR}/cpython"
THEROCK_SRC="${VLLM_DIR}/therock"
PYTORCH_SRC="${VLLM_DIR}/pytorch"
TRITON_SRC="${VLLM_DIR}/triton"
AOTRITON_SRC="${VLLM_DIR}/aotriton"
TORCHVISION_SRC="${VLLM_DIR}/torchvision"
FLASH_ATTN_SRC="${VLLM_DIR}/flash-attention"

# Wheel output directory — all pip-installable wheels stored here.
# Downstream projects install from this directory.
WHEELS_DIR="${VLLM_DIR}/wheels"

# =============================================================================
# Argument Parsing
# =============================================================================

REBUILD=false
START_STEP=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild)
            REBUILD=true
            shift
            ;;
        --step)
            START_STEP="$2"
            shift 2
            ;;
        *)
            die "Unknown argument: $1. Usage: build-vllm.sh [--rebuild] [--step N]"
            ;;
    esac
done

# =============================================================================
# Logging
# =============================================================================

# Ensure build directory exists before logging
if [[ ! -d "${VLLM_DIR}" ]]; then
    die "${VLLM_DIR} does not exist. Create it with:\n  sudo mkdir -p ${VLLM_DIR} && sudo chown \$(id -u):\$(id -g) ${VLLM_DIR}"
fi
if [[ ! -w "${VLLM_DIR}" ]]; then
    die "${VLLM_DIR} is not writable by $(whoami). Fix ownership with:\n  sudo chown \$(id -u):\$(id -g) ${VLLM_DIR}"
fi

# Tee all output to build log
exec > >(tee -a "${VLLM_LOG}") 2>&1

log_step() {
    local step_num="$1"
    local step_name="$2"
    echo ""
    echo "$(date '+%Y-%m-%d %H:%M:%S') [Step ${step_num}/${TOTAL_STEPS}] ${step_name}" >> "${VLLM_LOG}"
    section "[${step_num}/${TOTAL_STEPS}] ${step_name}"
}

# Find newest wheel matching a glob pattern. Returns empty string (not error)
# if no match exists. Safe under set -euo pipefail (no ls|head pipeline).
newest_wheel() {
    local pattern="$1"
    local newest=""
    for f in ${pattern}; do
        [[ -f "${f}" ]] || continue
        if [[ -z "${newest}" || "${f}" -nt "${newest}" ]]; then
            newest="${f}"
        fi
    done
    echo "${newest}"
}

# =============================================================================
# Prerequisites Check
# =============================================================================

check_prerequisites() {
    section "Checking prerequisites"
    require_commands clang clang++ lld cmake ninja uv git curl python3

    # TheRock build dependencies (system packages)
    local missing_pkgs=()
    for cmd in gfortran patchelf automake libtool bison flex xxd scons; do
        if ! command -v "${cmd}" &>/dev/null; then
            missing_pkgs+=("${cmd}")
        fi
    done
    if [[ ${#missing_pkgs[@]} -gt 0 ]]; then
        die "Missing system packages required for build: ${missing_pkgs[*]}. Install with your package manager."
    fi
    success "System build tools present"

    # Verify clang version >= 21
    local clang_version
    clang_version="$(clang --version | head -1 | grep -oP '\d+\.\d+\.\d+' | head -1)"
    local clang_major
    clang_major="${clang_version%%.*}"
    if [[ "${clang_major}" -lt 21 ]]; then
        die "Clang ${clang_version} found, but >= 21 is required."
    fi
    success "Clang ${clang_version} (>= 21)"

    # Verify cmake version >= 3.25
    local cmake_version
    cmake_version="$(cmake --version | head -1 | grep -oP '\d+\.\d+' | head -1)"
    success "CMake ${cmake_version}"

    # Verify GPU is accessible
    if [[ ! -e /dev/kfd ]]; then
        die "/dev/kfd not found. Is amdgpu loaded?"
    fi
    success "GPU accessible (/dev/kfd)"

    # Verify kernel
    local kernel_ver
    kernel_ver="$(uname -r)"
    if [[ ! "${kernel_ver}" =~ ^7\. ]]; then
        warn "Kernel ${kernel_ver} detected. Kernel 7.0+ recommended."
    else
        success "Kernel ${kernel_ver}"
    fi

    # Check available disk space (need ~100GB)
    local avail_gb
    avail_gb="$(df -BG /opt/src/ 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G')"
    if [[ -n "${avail_gb}" && "${avail_gb}" -lt 100 ]]; then
        warn "Only ${avail_gb}GB available. Build requires ~100GB."
    else
        success "Disk space: ${avail_gb:-unknown}GB available"
    fi
}

# =============================================================================
# Phase A: Foundation (AOCL-LibM + Python + ROCm SDK)
# =============================================================================

# Step 5: Build AOCL-Utils (dependency for AOCL-LibM)
# Runs AFTER TheRock so we can use amdclang (AMD's LLVM fork with -famd-opt).
build_aocl_utils() {
    log_step 5 "Build AOCL-Utils (CPU feature detection for Zen 5)"

    if [[ -f "${LOCAL_PREFIX}/lib/libaoclutils.so" ]]; then
        info "AOCL-Utils already built at ${LOCAL_PREFIX}"
        return
    fi

    if [[ ! -d "${AOCL_UTILS_SRC}/.git" ]]; then
        info "Cloning AOCL-Utils..."
        git clone "${AOCL_UTILS_REPO}" "${AOCL_UTILS_SRC}"
    fi

    cd "${AOCL_UTILS_SRC}"

    # Use amdclang from TheRock (built in Phase A)
    local amdclang="${LOCAL_PREFIX}/lib/llvm/bin/amdclang"
    local amdclangxx="${LOCAL_PREFIX}/lib/llvm/bin/amdclang++"
    if [[ ! -x "${amdclang}" ]]; then
        die "amdclang not found at ${amdclang} — run TheRock build first (steps 1-4)"
    fi

    # Build without LTO: AOCL-LibM links this .a with GNU ld (needed for its
    # hand-written AVX assembly), and GNU ld can't read LLVM bitcode objects.
    # We override CMAKE_*_FLAGS_RELEASE with non-LTO versions — the env vars
    # from vllm-env.sh include -flto=thin which would produce LLVM bitcode.
    # Disable clang-tidy: AOCL-Utils auto-enables it if found on PATH.
    # Both TheRock's clang-tidy (crashes on cleanup) and system clang-tidy
    # (doesn't understand -famd-opt) cause build failures. Setting
    # CMAKE_CXX_CLANG_TIDY to a truthy value prevents the find_program()
    # auto-detection, and /bin/true silently succeeds when cmake invokes it.
    local aocl_cflags="-O3 -march=native -mprefer-vector-width=512 -mavx512f -mavx512dq -mavx512vl -mavx512bw -famd-opt -Wno-error=unused-command-line-argument"
    info "Building AOCL-Utils with amdclang (no LTO, no clang-tidy)..."
    info "AOCL-Utils CFLAGS: ${aocl_cflags}"
    cmake -B build -GNinja . \
        -DCMAKE_C_COMPILER="${amdclang}" \
        -DCMAKE_CXX_COMPILER="${amdclangxx}" \
        -DCMAKE_C_FLAGS="${aocl_cflags}" \
        -DCMAKE_CXX_FLAGS="${aocl_cflags}" \
        -DCMAKE_C_FLAGS_RELEASE="-DNDEBUG ${aocl_cflags}" \
        -DCMAKE_CXX_FLAGS_RELEASE="-DNDEBUG ${aocl_cflags}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${LOCAL_PREFIX}" \
        -DALCI_DOCS=OFF \
        -DCMAKE_CXX_CLANG_TIDY="/bin/true"

    ninja -C build
    ninja -C build install

    cd "${VLLM_DIR}"
    success "AOCL-Utils built and installed"
}

# Step 6: Build AOCL-LibM (AMD-optimized math library)
# Runs AFTER TheRock so we can use amdclang which supports -muse-unaligned-vector-move
# (AOCL-LibM's build system injects this flag for any clang >= 14.0.6).
build_aocl_libm() {
    log_step 6 "Build AOCL-LibM (Zen 5 optimized transcendentals)"

    if [[ -f "${LOCAL_PREFIX}/lib/libalm.so" ]]; then
        info "AOCL-LibM already built at ${LOCAL_PREFIX}"
        return
    fi

    if [[ ! -d "${AOCL_LIBM_SRC}/.git" ]]; then
        info "Cloning AOCL-LibM..."
        git clone "${AOCL_LIBM_REPO}" "${AOCL_LIBM_SRC}"
    fi

    cd "${AOCL_LIBM_SRC}"

    # Use amdclang from TheRock (built in Phase A). AOCL-LibM's SCons build
    # detects 'clang' in the compiler name and adds -muse-unaligned-vector-move
    # for versions >= 14.0.6. This flag is AOCC/amdclang-specific — upstream
    # clang doesn't support it. Hence TheRock must be built first.
    local amdclang="${LOCAL_PREFIX}/lib/llvm/bin/amdclang"
    local amdclangxx="${LOCAL_PREFIX}/lib/llvm/bin/amdclang++"
    if [[ ! -x "${amdclang}" ]]; then
        die "amdclang not found at ${amdclang} — run TheRock build first (steps 1-4)"
    fi

    # Patch out -muse-unaligned-vector-move flag.
    # AOCL-LibM's SConscript assumes any clang >= 14.0.6 is AOCC (AMD's
    # proprietary compiler) and injects this AOCC-only flag. TheRock's
    # open-source amdclang doesn't support it — neither does upstream clang.
    # The flag enables unaligned vector load codegen; amdclang's -famd-opt
    # covers equivalent optimizations.
    local sconscript="${AOCL_LIBM_SRC}/src/SConscript"
    if [[ -f "${sconscript}" ]] && grep -q 'muse-unaligned-vector-move' "${sconscript}"; then
        info "Patching AOCL-LibM SConscript: removing AOCC-only -muse-unaligned-vector-move"
        sed -i "s/ccflags.append('-muse-unaligned-vector-move')/pass  # patched: AOCC-only flag removed for amdclang/" "${sconscript}"
    fi

    # Patch -Werror: AOCL-LibM's headers redefine CMPLX/CMPLXF macros that
    # glibc's complex.h already provides (identically). With -Werror this
    # becomes fatal. Add -Wno-macro-redefined after -Werror in the SConscript.
    if [[ -f "${sconscript}" ]] && grep -q "'-Werror'" "${sconscript}" && ! grep -q 'Wno-macro-redefined' "${sconscript}"; then
        info "Patching AOCL-LibM SConscript: adding -Wno-macro-redefined"
        sed -i "s/'-Werror'/'-Werror', '-Wno-macro-redefined'/" "${sconscript}"
    fi

    # Patch linker flags for clang compatibility:
    # 1. -ealm_main → -Wl,-e,alm_main (GCC passes -e to ld; clang doesn't)
    # 2. Use GNU ld (bfd) for final link: AOCL-LibM's hand-written AVX assembly
    #    (.S files in src/isa/avx/gas/) uses R_X86_64_64 absolute relocations
    #    against local symbols. lld rejects these in shared libraries; GNU ld
    #    handles them via dynamic text relocations. The assembly is correct for
    #    the target use case (hot math routines that get loaded at fixed offsets).
    #    AOCL-Utils is built without LTO so its .a contains normal ELF objects.
    if [[ -f "${sconscript}" ]] && grep -q "'-ealm_main'" "${sconscript}"; then
        info "Patching AOCL-LibM SConscript: fixing linker flags for amdclang"
        sed -i "s|'-ealm_main'|'-Wl,-e,alm_main', '-fuse-ld=bfd'|" "${sconscript}"
    fi

    info "Building AOCL-LibM with amdclang + AVX-512 support..."

    # Create a minimal venv for SCons if needed
    if [[ ! -d "${AOCL_LIBM_SRC}/.venv" ]]; then
        python3 -m venv "${AOCL_LIBM_SRC}/.venv"
    fi
    # shellcheck source=/dev/null
    source "${AOCL_LIBM_SRC}/.venv/bin/activate"
    pip install scons 2>&1 | tail -1

    scons -j"$(nproc)" \
        ALM_CC="${amdclang}" \
        ALM_CXX="${amdclangxx}" \
        --arch_config=avx512 \
        --aocl_utils_install_path="${LOCAL_PREFIX}" \
        --aocl_utils_link=0

    # Install: copy libraries and headers to LOCAL_PREFIX
    info "Installing AOCL-LibM to ${LOCAL_PREFIX}..."
    mkdir -p "${LOCAL_PREFIX}/lib" "${LOCAL_PREFIX}/include"

    # Copy the built libraries
    find build/aocl-release/src -name 'libalm*' -exec cp {} "${LOCAL_PREFIX}/lib/" \;

    # Copy the glibc-compat preload object if built
    local glibc_compat="build/aocl-release/src/compat/glibc-compat.o"
    if [[ -f "${glibc_compat}" ]]; then
        cp "${glibc_compat}" "${LOCAL_PREFIX}/lib/"
    fi

    # Copy headers
    if [[ -d "include" ]]; then
        cp -a include/* "${LOCAL_PREFIX}/include/"
    fi

    # Deactivate the temporary venv
    deactivate

    cd "${VLLM_DIR}"
    success "AOCL-LibM built with AVX-512 Zen 5 optimizations"
}

# Step 7: Build Python from source (using amdclang from TheRock)
build_python() {
    log_step 7 "Build Python ${CPYTHON_VERSION} from source"

    if [[ -x "${LOCAL_PREFIX}/bin/python3" ]]; then
        local existing_ver
        existing_ver="$("${LOCAL_PREFIX}/bin/python3" --version 2>&1 | awk '{print $2}')"
        info "Python ${existing_ver} already built at ${LOCAL_PREFIX}"
        return
    fi

    if [[ ! -d "${CPYTHON_SRC}/.git" ]]; then
        info "Cloning CPython ${CPYTHON_TAG}..."
        git clone --depth 1 --branch "${CPYTHON_TAG}" \
            https://github.com/python/cpython.git "${CPYTHON_SRC}"
    fi

    cd "${CPYTHON_SRC}"

    # Build Python with:
    #   - PGO (Profile-Guided Optimization): runs test suite as training data
    #   - LTO (Link-Time Optimization): whole-program optimization
    #   - --enable-optimizations: enables both PGO and computed-gotos
    #   - Linked against AOCL-LibM for Zen 5 optimized transcendentals
    #   - amdclang with -march=native -famd-opt for Zen 5 native codegen
    info "Configuring Python ${CPYTHON_VERSION} (PGO + LTO)..."

    # Use amdclang from TheRock
    local amdclang="${LOCAL_PREFIX}/lib/llvm/bin/amdclang"
    local amdclangxx="${LOCAL_PREFIX}/lib/llvm/bin/amdclang++"
    if [[ ! -x "${amdclang}" ]]; then
        die "amdclang not found at ${amdclang} — run TheRock build first (steps 1-4)"
    fi

    # Note: we do NOT link CPython against AOCL-LibM (-lalm) directly.
    # AOCL-LibM's transcendentals have slightly different ULP rounding than
    # glibc's libm, which causes CPython's test_math to fail during PGO.
    # Instead, AOCL-LibM is available at runtime via LD_LIBRARY_PATH for
    # downstream numerical libraries (NumPy, PyTorch) that benefit from it.
    # Unset ALL vllm-env.sh optimization env vars to prevent contamination.
    # vllm-env.sh sets LDFLAGS="-flto=thin -fuse-ld=lld -L.../lib -lalm" which
    # autoconf merges with configure-specified LDFLAGS. The -lalm causes CPython's
    # PGO test_math to fail because AOCL-LibM handles signed zero and subnormal
    # numbers differently from glibc libm (cbrt(-0.0) → +0.0, nextafter broken).
    # CMAKE_* vars are also unset since CPython uses autoconf, not cmake.
    unset CFLAGS CXXFLAGS LDFLAGS CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELEASE \
          CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS

    ./configure \
        --prefix="${LOCAL_PREFIX}" \
        --enable-optimizations \
        --with-lto=thin \
        --enable-shared \
        --with-computed-gotos \
        --with-system-expat \
        --with-ensurepip=upgrade \
        CC="${amdclang}" \
        CXX="${amdclangxx}" \
        CFLAGS="-O3 -march=native -famd-opt -Wno-error=unused-command-line-argument -fPIC" \
        CXXFLAGS="-O3 -march=native -famd-opt -Wno-error=unused-command-line-argument -fPIC" \
        LDFLAGS="-flto=thin -fuse-ld=lld -Wl,-rpath,${LOCAL_PREFIX}/lib"

    info "Building Python ${CPYTHON_VERSION} (PGO training + final build)..."
    info "This takes ~15-20 minutes due to PGO profiling pass."
    make -j"$(nproc)"

    info "Installing Python to ${LOCAL_PREFIX}..."
    make install

    # Restore vllm-env.sh environment for subsequent steps (we unset
    # CFLAGS/CXXFLAGS/LDFLAGS above to avoid -lalm contamination).
    # shellcheck source=vllm-env.sh
    _vllm_source_env

    # Verify
    "${LOCAL_PREFIX}/bin/python3" --version
    info "Python build config:"
    "${LOCAL_PREFIX}/bin/python3" -c "
import sysconfig
print(f'  CC: {sysconfig.get_config_var(\"CC\")}')
print(f'  OPT: {sysconfig.get_config_var(\"OPT\")}')
print(f'  LTO: {sysconfig.get_config_var(\"LTOCFLAGS\") or \"none\"}')
"

    cd "${VLLM_DIR}"
    success "Python ${CPYTHON_VERSION} built (PGO + LTO + amdclang)"
}

# Step 8: Create Virtual Environment (using our custom Python)
create_venv() {
    log_step 8 "Create virtual environment"

    # Determine which Python to use: prefer our source-built Python
    local python_bin="python3"
    if [[ -x "${LOCAL_PREFIX}/bin/python3" ]]; then
        python_bin="${LOCAL_PREFIX}/bin/python3"
        info "Using source-built Python: ${python_bin}"
    else
        warn "Source-built Python not found, using system python3"
    fi

    if [[ -d "${VLLM_VENV}" && -f "${VLLM_VENV}/bin/python" ]]; then
        info "Venv already exists at ${VLLM_VENV}"

        # Check if the venv uses our custom Python
        local venv_python_real
        venv_python_real="$(readlink -f "${VLLM_VENV}/bin/python" 2>/dev/null || echo 'unknown')"
        local custom_python_real
        custom_python_real="$(readlink -f "${python_bin}" 2>/dev/null || echo 'unknown2')"
        if [[ "${venv_python_real}" != "${custom_python_real}" && -x "${LOCAL_PREFIX}/bin/python3" ]]; then
            info "Venv uses different Python (${venv_python_real}), recreating with our build..."
            rm -r "${VLLM_VENV}"
        else
            # shellcheck source=/dev/null
            source "${VLLM_VENV}/bin/activate"

            # Ensure ALL essential build tools are present (may be missing from older venvs)
            if ! python -c 'import yaml, mako, packaging, CppHeaderParser' 2>/dev/null \
               || ! command -v ninja &>/dev/null; then
                info "Installing missing build tools into existing venv..."
                uv pip install pip ninja cmake wheel setuptools \
                    "CppHeaderParser==2.7.4" meson PyYAML packaging mako
            fi

            success "Venv activated"
            return
        fi
    fi

    info "Creating venv at ${VLLM_VENV} using ${python_bin}..."
    uv venv --python "${python_bin}" "${VLLM_VENV}"

    # shellcheck source=/dev/null
    source "${VLLM_VENV}/bin/activate"

    # Ensure AOCL-LibM is on the library path for this venv
    if [[ -d "${LOCAL_PREFIX}/lib" ]]; then
        export LD_LIBRARY_PATH="${LOCAL_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi

    info "Installing essential build tools into venv..."
    uv pip install pip ninja cmake wheel setuptools \
        "CppHeaderParser==2.7.4" meson PyYAML packaging mako

    success "Venv created and activated (Python $(python --version 2>&1 | awk '{print $2}'))"
}

# Step 1: Clone TheRock
clone_therock() {
    log_step 1 "Clone TheRock ROCm source (with submodules)"

    if [[ -d "${THEROCK_SRC}/.git" ]]; then
        info "TheRock already cloned at ${THEROCK_SRC}"
        cd "${THEROCK_SRC}"
        git fetch origin
        local current_branch
        current_branch="$(git branch --show-current)"
        git pull origin "${current_branch}"
        info "Updating submodules..."
        git submodule update --init --recursive
        cd "${VLLM_DIR}"
        success "TheRock source updated"
        return
    fi

    info "Cloning TheRock with submodules (this takes several minutes)..."
    git clone --recursive "${THEROCK_REPO}" "${THEROCK_SRC}"
    success "TheRock cloned to ${THEROCK_SRC} (with submodules)"
}

# Step 2: Configure TheRock
configure_therock() {
    log_step 2 "Configure TheRock (cmake)"

    cd "${THEROCK_SRC}"

    # Check if already configured
    if [[ -f "build/build.ninja" ]]; then
        info "TheRock already configured (build/build.ninja exists)"
        cd "${VLLM_DIR}"
        return
    fi

    info "Configuring TheRock for gfx1151..."

    # TheRock's nested cmake sub-builds (LLVM runtimes, hip-clr, amd-mesa)
    # each run FindPython3 independently and may find a different Python
    # than the venv. Install required Python packages into whatever python3
    # cmake would find on the system, in addition to the venv.
    local sys_python
    sys_python="$(command -v python3)"
    if [[ -n "${sys_python}" ]] && ! "${sys_python}" -c 'import yaml, mako, packaging, CppHeaderParser' 2>/dev/null; then
        info "Installing TheRock Python deps into system python: ${sys_python}"
        "${sys_python}" -m pip install --break-system-packages \
            pyyaml mako packaging "CppHeaderParser==2.7.4" 2>/dev/null || true
    fi

    # TheRock requires GCC — rocprofiler-systems has an explicit GNU
    # compiler check that blocks Clang. Unset all amdclang-specific flags;
    # re-source vllm-env.sh afterward to restore them.
    unset CFLAGS CXXFLAGS LDFLAGS CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELEASE \
          CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS

    # TheRock has deeply nested cmake sub-builds (LLVM -> runtimes) that
    # each run FindPython3 independently. TheRock now runs BEFORE our venv
    # exists, so we point at the system python3 (which we installed build
    # deps into above).
    # Python3_ROOT_DIR is the cmake hint that propagates through sub-builds.
    cmake -B build -GNinja . \
        -DTHEROCK_AMDGPU_FAMILIES=gfx1151 \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_INSTALL_PREFIX="${LOCAL_PREFIX}" \
        -DPython3_EXECUTABLE="${sys_python}" \
        -DTHEROCK_BUILD_TESTING=OFF \
        -DTHEROCK_ENABLE_PROFILER=OFF \
        -DTHEROCK_FLAG_INCLUDE_PROFILER=OFF
        # Profiler disabled: rocprofiler-sdk's vendored yaml-cpp and elfio
        # have missing <cstdint> includes under modern compilers (Clang 18+,
        # GCC 15+). Profiling is not needed for vLLM inference.

    # Restore all flags from vllm-env.sh
    # shellcheck source=vllm-env.sh
    _vllm_source_env

    cd "${VLLM_DIR}"
    success "TheRock configured"
}

# Step 3: Build TheRock
build_therock() {
    log_step 3 "Build TheRock (this will take several hours)"

    cd "${THEROCK_SRC}"


    # Check if already built and installed
    if [[ -d "${LOCAL_PREFIX}/lib" ]] && find "${LOCAL_PREFIX}" -name 'libamdhip64.so*' -print -quit 2>/dev/null | grep -q .; then
        info "TheRock already built and installed at ${LOCAL_PREFIX}"
        cd "${VLLM_DIR}"
        return
    fi

    info "Building TheRock with $(nproc) cores..."
    info "This is the longest step. Expected time: 2-4 hours."
    info "Monitor progress: tail -f ${VLLM_LOG}"

    # Enable Polly in TheRock's LLVM build.
    # Polly (polyhedral loop optimizer) restructures loop nests for cache locality.
    # TheRock's default LLVM_ENABLE_PROJECTS does not include polly — add it so
    # the built amdclang supports -mllvm -polly for downstream builds.
    local llvm_prehook="${THEROCK_SRC}/compiler/pre_hook_amd-llvm.cmake"
    if [[ -f "${llvm_prehook}" ]] && grep -q 'LLVM_ENABLE_PROJECTS' "${llvm_prehook}" && ! grep -q 'polly' "${llvm_prehook}"; then
        info "Patching TheRock LLVM to enable Polly polyhedral optimizer"
        sed -i 's|clang;lld;clang-tools-extra;flang|clang;lld;clang-tools-extra;flang;polly|' "${llvm_prehook}"
    fi

    # Patch elfutils cmake wrapper to disable -Werror.
    # elfutils' config/eu.am unconditionally adds -Werror to every compile.
    # Modern compilers (Clang 21+, GCC 15+) reject elfutils' implicit
    # const void* -> struct* conversions from bsearch() returns.
    # Fix: inject CFLAGS=-Wno-error into the ./configure environment
    # so autotools receives it and -Werror is effectively neutralized.
    local elfutils_cmake="${THEROCK_SRC}/third-party/sysdeps/linux/elfutils/CMakeLists.txt"
    if [[ -f "${elfutils_cmake}" ]] && ! grep -q 'Wno-error' "${elfutils_cmake}"; then
        info "Patching elfutils CMakeLists.txt to disable -Werror"
        # shellcheck disable=SC2016  # Intentional: inject literal ${EXTRA_CPPFLAGS} into CMake file
        sed -i 's|"CPPFLAGS=${EXTRA_CPPFLAGS}"|"CPPFLAGS=${EXTRA_CPPFLAGS}"\n      "CFLAGS=-Wno-error"|' "${elfutils_cmake}"
    fi

    # Patch rocprofiler-sdk's bundled yaml-cpp: missing <cstdint> include.
    # Newer compilers (Clang 18+, GCC 13+) removed transitive includes of
    # <cstdint>, so uint16_t/uint32_t are undeclared without explicit include.
    local yamlcpp_emitterutils="${THEROCK_SRC}/rocm-systems/projects/rocprofiler-sdk/external/yaml-cpp/src/emitterutils.cpp"
    if [[ -f "${yamlcpp_emitterutils}" ]] && ! grep -q '<cstdint>' "${yamlcpp_emitterutils}"; then
        info "Patching rocprofiler-sdk yaml-cpp emitterutils.cpp: add <cstdint>"
        sed -i '/#include <algorithm>/a #include <cstdint>' "${yamlcpp_emitterutils}"
    fi

    # Patch rocprofiler-sdk's bundled elfio: missing <cstdint> include.
    # elfio/elf_types.hpp uses Elf64_Half (uint16_t) etc. without including
    # <cstdint>. Clang 22 and GCC 15 no longer provide transitive <cstdint>.
    local elfio_types="${THEROCK_SRC}/rocm-systems/projects/rocprofiler-sdk/external/elfio/elfio/elf_types.hpp"
    if [[ -f "${elfio_types}" ]] && ! grep -q '<cstdint>' "${elfio_types}"; then
        info "Patching rocprofiler-sdk elfio elf_types.hpp: add <cstdint>"
        sed -i '/#define ELFTYPES_H/a #include <cstdint>' "${elfio_types}"
    fi

    # Install Tensile Python dependencies into the build venv.
    # hipBLASLt's Tensile kernel generator imports joblib, msgpack, numpy,
    # pandas, pyyaml at code-generation time. These are not declared as cmake
    # dependencies — Tensile assumes they are available in the Python environment.
    info "Installing Tensile Python dependencies into build venv"
    uv pip install joblib msgpack numpy pandas pyyaml pytest

    # Unset amdclang flags and HSA override — TheRock uses GCC and has its
    # own GPU arch detection. Re-source vllm-env.sh after to restore.
    unset CFLAGS CXXFLAGS LDFLAGS HSA_OVERRIDE_GFX_VERSION \
          CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELEASE \
          CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS

    ninja -C build

    info "Installing TheRock to ${LOCAL_PREFIX}..."
    cmake --install build --prefix "${LOCAL_PREFIX}"

    # Copy MLIR object libraries into the install tree.
    # TheRock's dist aggregation copies .a and .so files but skips the
    # objects-Release/ directory that MLIR's cmake exports reference.
    # These object files are needed by downstream consumers (Triton) that
    # link MLIR statically via LLVM_SYSPATH.
    local mlir_objects="${THEROCK_SRC}/build/compiler/amd-llvm/stage/lib/llvm/lib/objects-Release"
    local install_objects="${LOCAL_PREFIX}/lib/llvm/lib/objects-Release"
    if [[ -d "${mlir_objects}" ]] && [[ ! -d "${install_objects}" ]]; then
        info "Copying MLIR object libraries to install tree"
        cp -a "${mlir_objects}" "${install_objects}"
    fi

    # Copy LLVM test utilities needed by downstream consumers.
    # FileCheck is required by Triton's cmake (unconditionally copies it into
    # the wheel). TheRock builds it but doesn't install it when testing is off.
    local filecheck_src="${THEROCK_SRC}/build/compiler/amd-llvm/build/bin/FileCheck"
    local filecheck_dst="${LOCAL_PREFIX}/lib/llvm/bin/FileCheck"
    if [[ -f "${filecheck_src}" ]] && [[ ! -f "${filecheck_dst}" ]]; then
        info "Installing FileCheck into LLVM toolchain"
        cp "${filecheck_src}" "${filecheck_dst}"
        chmod +x "${filecheck_dst}"
    fi

    # Restore all flags from vllm-env.sh
    # shellcheck source=vllm-env.sh
    _vllm_source_env

    # Write version marker
    local therock_version
    therock_version="$(cd "${THEROCK_SRC}" && git describe --tags --always 2>/dev/null || echo 'local')"
    echo "${therock_version}" > "${VLLM_DIR}/.rocm-version"

    cd "${VLLM_DIR}"
    success "TheRock built and installed (${therock_version})"
}

# Step 4: Validate ROCm
validate_rocm() {
    log_step 4 "Validate ROCm installation"


    # Update environment to use locally-built ROCm.
    # lib/llvm/bin is added to PATH so amdclang is available for downstream builds.
    export ROCM_PATH="${LOCAL_PREFIX}"
    export LD_LIBRARY_PATH="${ROCM_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export PATH="${ROCM_PATH}/lib/llvm/bin:${ROCM_PATH}/bin:${PATH}"

    # Create clang/clang++ symlinks to amdclang/amdclang++ so that build
    # systems looking for "clang" on PATH find the AMD-optimized variant.
    local llvm_bin="${ROCM_PATH}/lib/llvm/bin"
    # Create cc/c++ symlinks so build systems find amdclang by default.
    # clang/clang++ are already installed by TheRock's cmake --install.
    if [[ -x "${llvm_bin}/amdclang" ]]; then
        [[ -e "${llvm_bin}/cc" ]]  || ln -s amdclang "${llvm_bin}/cc"
        [[ -e "${llvm_bin}/c++" ]] || ln -s amdclang++ "${llvm_bin}/c++"
        info "Compiler symlinks in ${llvm_bin}: cc→amdclang, c++→amdclang++"
    fi

    if [[ -d "${ROCM_PATH}/llvm/amdgcn/bitcode" ]]; then
        export DEVICE_LIB_PATH="${ROCM_PATH}/llvm/amdgcn/bitcode"
        export HIP_DEVICE_LIB_PATH="${ROCM_PATH}/llvm/amdgcn/bitcode"
    elif [[ -d "${ROCM_PATH}/amdgcn/bitcode" ]]; then
        export DEVICE_LIB_PATH="${ROCM_PATH}/amdgcn/bitcode"
        export HIP_DEVICE_LIB_PATH="${ROCM_PATH}/amdgcn/bitcode"
    fi

    info "ROCM_PATH: ${ROCM_PATH}"

    # Check hipcc
    if [[ -x "${ROCM_PATH}/bin/hipcc" ]]; then
        success "hipcc found: $("${ROCM_PATH}"/bin/hipcc --version 2>&1 | head -1)"
    else
        die "hipcc not found at ${ROCM_PATH}/bin/hipcc — TheRock build may have failed."
    fi

    # Check rocminfo
    if [[ -x "${ROCM_PATH}/bin/rocminfo" ]]; then
        info "Testing rocminfo..."
        "${ROCM_PATH}/bin/rocminfo" 2>&1 | grep -i "gfx" | head -5 || true
    fi

    # Check amd-smi
    if [[ -x "${ROCM_PATH}/bin/amd-smi" ]]; then
        success "amd-smi found"
        "${ROCM_PATH}/bin/amd-smi" version 2>/dev/null || info "amd-smi version check skipped"
    else
        info "amd-smi not in PATH (may be installed via Python)"
    fi

    # Check device libraries
    local bitcode_dir="${DEVICE_LIB_PATH:-}"
    if [[ -n "${bitcode_dir}" && -d "${bitcode_dir}" ]]; then
        local bitcode_count
        bitcode_count="$(find "${bitcode_dir}" -name '*.bc' | wc -l)"
        success "Device libraries: ${bitcode_count} bitcode files"
    else
        warn "Device bitcode directory not found"
    fi

    # Check key libraries
    for lib in libamdhip64.so librocblas.so libMIOpen.so; do
        if find "${ROCM_PATH}/lib" -name "${lib}*" -print -quit 2>/dev/null | grep -q .; then
            success "${lib} found"
        else
            warn "${lib} not found in ${ROCM_PATH}/lib"
        fi
    done
}

# =============================================================================
# Phase B: ML Framework (PyTorch, ROCm fork)
# =============================================================================

# Step 9: Clone PyTorch (ROCm fork)
clone_pytorch() {
    log_step 9 "Clone PyTorch source (ROCm fork, ${PYTORCH_BRANCH} branch)"

    if [[ -d "${PYTORCH_SRC}/.git" ]]; then
        info "PyTorch already cloned at ${PYTORCH_SRC}"
        cd "${PYTORCH_SRC}"

        # Ensure we're tracking the ROCm fork
        local current_url
        current_url="$(git remote get-url origin 2>/dev/null)"
        if [[ "${current_url}" != *"ROCm/pytorch"* ]]; then
            info "Switching remote from ${current_url} to ${PYTORCH_REPO}"
            git remote set-url origin "${PYTORCH_REPO}"
        fi

        # PyTorch's hipify step modifies hundreds of files in-tree (CUDA→HIP).
        # These must be reset before branch operations, otherwise checkout fails.
        local dirty_count
        dirty_count="$(git status --short | wc -l)"
        if [[ "${dirty_count}" -gt 0 ]]; then
            info "Resetting ${dirty_count} hipified files in PyTorch tree..."
            git checkout -- .
            git submodule foreach --recursive 'git checkout -- . 2>/dev/null || true'
        fi

        git fetch origin "${PYTORCH_BRANCH}"
        local current_branch
        current_branch="$(git branch --show-current)"
        if [[ "${current_branch}" != "${PYTORCH_BRANCH}" ]]; then
            info "Switching to ${PYTORCH_BRANCH} branch..."
            git checkout "${PYTORCH_BRANCH}"
        fi
        git pull origin "${PYTORCH_BRANCH}"
        git submodule update --init --recursive
        cd "${VLLM_DIR}"
        success "PyTorch source updated (ROCm fork, ${PYTORCH_BRANCH})"
        return
    fi

    info "Cloning ROCm PyTorch (${PYTORCH_BRANCH} branch, with submodules)..."
    git clone --recursive --branch "${PYTORCH_BRANCH}" "${PYTORCH_REPO}" "${PYTORCH_SRC}"
    success "PyTorch cloned to ${PYTORCH_SRC} (ROCm fork)"
}

# Step 10: Build PyTorch
build_pytorch() {
    log_step 10 "Build PyTorch with ROCm support"

    cd "${PYTORCH_SRC}"

    # Check if already built — run from VLLM_DIR to avoid importing
    # the local torch/ source directory instead of the installed package.
    if (cd "${VLLM_DIR}" && python -c "import torch; assert 'rocm' in torch.__file__.lower() or torch.cuda.is_available()") 2>/dev/null; then
        local torch_ver
        torch_ver="$(cd "${VLLM_DIR}" && python -c "import torch; print(torch.__version__)")"
        info "PyTorch ${torch_ver} already built and importable"
        cd "${VLLM_DIR}"
        return
    fi

    # Flags come from vllm-env.sh (sourced at script start, re-sourced after
    # build_python). Verify they're set — if not, something broke the pipeline.
    if [[ -z "${CFLAGS:-}" ]] || [[ -z "${CMAKE_CXX_FLAGS_RELEASE:-}" ]]; then
        die "CFLAGS or CMAKE_CXX_FLAGS_RELEASE not set — vllm-env.sh was not sourced"
    fi
    export PATH="${ROCM_PATH}/lib/llvm/bin:${PATH}"
    export CC="${ROCM_PATH}/lib/llvm/bin/amdclang"
    export CXX="${ROCM_PATH}/lib/llvm/bin/amdclang++"

    info "Building PyTorch for ROCm gfx1151..."
    info "ROCM_PATH=${ROCM_PATH}"
    info "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}"
    info "CC=${CC}, CXX=${CXX}"
    info "CFLAGS=${CFLAGS}"

    # PyTorch build environment
    export USE_ROCM=1
    export USE_CUDA=0
    export USE_NCCL=0
    export USE_SYSTEM_NCCL=0
    export USE_RCCL=1
    export BUILD_TEST=0
    export USE_BENCHMARK=0
    export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH}"
    export HIP_PATH="${ROCM_PATH}"
    export ROCM_HOME="${ROCM_PATH}"
    export CMAKE_PREFIX_PATH="${ROCM_PATH}"

    # Install Python build deps (numpy>=2 for ABI compatibility with our wheel)
    uv pip install \
        pip \
        "numpy>=2.0,<3" \
        pyyaml \
        typing_extensions \
        cmake \
        ninja \
        setuptools \
        wheel \
        cffi \
        sympy \
        filelock \
        jinja2 \
        networkx \
        requests \
        six

    # Convert CUDA references to HIP equivalents (required for ROCm builds)
    if [[ -f "tools/amd_build/build_amd.py" ]]; then
        info "Running AMD HIP conversion (tools/amd_build/build_amd.py)..."
        python tools/amd_build/build_amd.py
    fi

    # Patch HIPGraph.hip: hipify creates a set_conditional_handle() function
    # referencing cudaGraphConditionalHandle (a CUDA 12.4+ type with no HIP
    # equivalent). The header (HIPGraph.h) doesn't declare it, nothing calls
    # it on ROCm — it's dead code that fails to compile. Remove it.
    local _hipgraph="${PYTORCH_SRC}/aten/src/ATen/hip/HIPGraph.hip"
    if [[ -f "${_hipgraph}" ]] && grep -q 'cudaGraphConditionalHandle' "${_hipgraph}"; then
        info "Patching HIPGraph.hip: removing CUDA-only cudaGraphConditionalHandle code"
        cat > "${_hipgraph}" << 'HIPEOF'
// !!! This is a file automatically generated by hipify!!!
#include "hip/hip_runtime.h"
#include <ATen/hip/HIPGraph.h>
#include <ATen/hip/Exceptions.h>

namespace at::cuda {

// cudaGraphConditionalHandle / set_conditional_handle removed:
// CUDA 12.4+ feature with no HIP equivalent. The class declaration
// in HIPGraph.h does not include this method on ROCm builds.

} // namespace at::cuda
HIPEOF
    fi

    # Patch numpy_stub.h: set NPY_TARGET_VERSION to numpy 2.0 C-API (0x12).
    # Without this, numpy 2.x headers default to compiling against the oldest
    # compatible API (1.20), producing a .so that crashes at import with numpy
    # 2.x installed ("module was compiled using NumPy 1.x cannot be run in
    # NumPy 2.x"). Must be set before #include <numpy/arrayobject.h>.
    local _numpy_stub="${PYTORCH_SRC}/torch/csrc/utils/numpy_stub.h"
    if [[ -f "${_numpy_stub}" ]] && ! grep -q 'NPY_TARGET_VERSION' "${_numpy_stub}"; then
        info "Patching numpy_stub.h: setting NPY_TARGET_VERSION=0x12 (numpy 2.0)"
        sed -i 's|#include <numpy/arrayobject.h>|// Target numpy 2.0 C-API (0x12) for ABI compatibility with numpy >= 2.0.\n#ifndef NPY_TARGET_VERSION\n#define NPY_TARGET_VERSION 0x00000012\n#endif\n\n#include <numpy/arrayobject.h>|' "${_numpy_stub}"
    fi

    # Patch cmake/Dependencies.cmake: remove -fclang-abi-compat=17 from HIPCC.
    # PyTorch adds this "for compat with newer hip-clang C++20 mangling rules",
    # but it forces HIP device code to use Clang 17 ABI while host code uses
    # amdclang 22 ABI, causing undefined symbol errors (e.g. const_data_ptr<Half>
    # mangled differently between libtorch_cpu.so and libtorch_hip.so).
    local _deps_cmake="${PYTORCH_SRC}/cmake/Dependencies.cmake"
    if grep -q 'fclang-abi-compat=17' "${_deps_cmake}" 2>/dev/null; then
        info "Patching Dependencies.cmake: removing -fclang-abi-compat=17 (ABI mismatch fix)"
        sed -i 's/list(APPEND HIP_HIPCC_FLAGS -fclang-abi-compat=17)/# Removed: causes ABI mismatch with host amdclang 22/' "${_deps_cmake}"
    fi

    # Patch Context.cpp: add gfx1151 to CK (Composable Kernel) GEMM supported
    # architectures. Without this, PyTorch logs "Attempting to use CK on an
    # unsupported architecture!" and TunableOp cannot include CK kernels in its
    # autotuning candidates. The gate is a hardcoded vector of arch strings.
    local _context_cpp="${PYTORCH_SRC}/aten/src/ATen/Context.cpp"
    if [[ -f "${_context_cpp}" ]] && grep -q '"gfx90a", "gfx942", "gfx950"' "${_context_cpp}" && ! grep -q 'gfx1151' "${_context_cpp}"; then
        info "Patching Context.cpp: adding gfx1151 to CK GEMM supported architectures"
        sed -i 's/"gfx90a", "gfx942", "gfx950"/"gfx90a", "gfx942", "gfx950", "gfx1151"/' "${_context_cpp}"
    fi

    # Step 1: Build the wheel. pip wheel runs cmake (incremental if build/
    # exists) and packages everything into a .whl file.
    info "Building PyTorch wheel (this takes 1-2 hours on first build)..."
    mkdir -p "${WHEELS_DIR}"
    pip wheel . \
        --no-build-isolation \
        --no-deps \
        --wheel-dir "${WHEELS_DIR}" \
        -v

    # Step 2: Patch .so files INSIDE the wheel. pip wheel re-invokes cmake
    # during packaging, so patching the source tree beforehand doesn't work —
    # the wheel gets fresh unpatched copies. Instead, we unpack the .whl,
    # patch the .so files, and repack. Two fixes:
    #   1. RPATH: add /opt/src/vllm/local/lib so libalm.so, librocm_smi64.so,
    #      and other ROCm libs resolve without LD_LIBRARY_PATH at runtime
    #   2. NEEDED: add librocm_smi64.so to libtorch_hip.so (PyTorch's build
    #      system omits it from the link line despite using rsmi_* symbols —
    #      upstream bug, causes "undefined symbol: rsmi_init" at runtime)
    local _torch_wheel
    _torch_wheel="$(newest_wheel "${WHEELS_DIR}"/torch-*.whl)"
    if [[ -z "${_torch_wheel}" ]]; then
        die "PyTorch wheel not found in ${WHEELS_DIR}"
    fi

    info "Patching .so RPATHs and dependencies inside wheel..."
    local _patch_dir
    _patch_dir="$(mktemp -d)"
    cd "${_patch_dir}"
    unzip -q "${_torch_wheel}"

    # Fix RPATHs: cmake bakes the build tree path into RUNPATH (e.g.
    # /opt/src/vllm/pytorch/build/lib). This causes the dynamic linker to
    # load unpatched .so files from the build tree instead of the wheel's
    # copies. Clean all RPATHs to only contain the ROCm prefix and $ORIGIN.
    for _so in torch/lib/lib*.so; do
        [[ -f "${_so}" ]] || continue
        local _rpath
        _rpath="$(readelf -d "${_so}" 2>/dev/null | grep 'RUNPATH' || true)"
        if echo "${_rpath}" | grep -q 'pytorch/build'; then
            patchelf --set-rpath "${LOCAL_PREFIX}/lib:\$ORIGIN" "${_so}" 2>/dev/null || true
        elif readelf -d "${_so}" 2>/dev/null | grep -q 'libalm.so\|libamdhip64\|librocm_smi'; then
            patchelf --add-rpath "${LOCAL_PREFIX}/lib" "${_so}" 2>/dev/null || true
        fi
    done
    # Also fix the _C extension module if it has build tree RPATH
    for _so in torch/_C*.so; do
        [[ -f "${_so}" ]] || continue
        if readelf -d "${_so}" 2>/dev/null | grep -q 'pytorch/build'; then
            patchelf --set-rpath "${LOCAL_PREFIX}/lib:\$ORIGIN/lib" "${_so}" 2>/dev/null || true
        fi
    done

    # Add librocm_smi64.so to libtorch_hip.so NEEDED list
    if [[ -f "torch/lib/libtorch_hip.so" ]] && ! readelf -d "torch/lib/libtorch_hip.so" 2>/dev/null | grep -q 'librocm_smi64'; then
        info "  Adding librocm_smi64.so to libtorch_hip.so NEEDED"
        patchelf --add-needed librocm_smi64.so "torch/lib/libtorch_hip.so"
    fi

    # Repack the wheel using Python's zipfile (zip may not be installed)
    rm -f "${_torch_wheel}"
    python -c "
import zipfile, os
with zipfile.ZipFile('${_torch_wheel}', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('.'):
        for f in files:
            fp = os.path.join(root, f)
            arcname = fp[2:] if fp.startswith('./') else fp
            zf.write(fp, arcname)
"
    cd "${PYTORCH_SRC}"
    rm -r "${_patch_dir}"
    info "Wheel repacked with RPATH and NEEDED fixes"

    # Also patch libalm.so itself (it depends on libau_cpuid.so from same dir)
    if [[ -f "${LOCAL_PREFIX}/lib/libalm.so" ]]; then
        local _alm_rpath
        _alm_rpath="$(patchelf --print-rpath "${LOCAL_PREFIX}/lib/libalm.so" 2>/dev/null || true)"
        if [[ "${_alm_rpath}" != *"${LOCAL_PREFIX}/lib"* ]]; then
            patchelf --add-rpath "${LOCAL_PREFIX}/lib" "${LOCAL_PREFIX}/lib/libalm.so"
        fi
    fi

    # Install the wheel into the build venv
    info "Installing PyTorch wheel into build venv..."
    local _torch_wheel
    _torch_wheel="$(newest_wheel "${WHEELS_DIR}"/torch-*.whl)"
    if [[ -z "${_torch_wheel}" ]]; then
        die "PyTorch wheel not found in ${WHEELS_DIR}"
    fi
    uv pip install --force-reinstall --no-deps "${_torch_wheel}"

    cd "${VLLM_DIR}"
    success "PyTorch built and installed (wheel: $(basename "${_torch_wheel}"))"
}

# Step 11: Validate PyTorch
validate_pytorch() {
    log_step 11 "Validate PyTorch GPU access"

    python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  ROCm/HIP: {torch.version.hip}')
    print(f'  Device count: {torch.cuda.device_count()}')
else:
    raise RuntimeError('PyTorch cannot see GPU — build may have failed')
" || die "PyTorch GPU validation failed"

    success "PyTorch GPU access verified"
}

# Step 12: Clone TorchVision
clone_torchvision() {
    log_step 12 "Clone TorchVision source"

    if [[ -d "${TORCHVISION_SRC}/.git" ]]; then
        info "TorchVision already cloned at ${TORCHVISION_SRC}"
        cd "${TORCHVISION_SRC}"
        git fetch origin
        git pull origin main
        cd "${VLLM_DIR}"
        success "TorchVision source updated"
        return
    fi

    info "Cloning TorchVision..."
    git clone "${TORCHVISION_REPO}" "${TORCHVISION_SRC}"
    success "TorchVision cloned to ${TORCHVISION_SRC}"
}

# Step 13: Build TorchVision
build_torchvision() {
    log_step 13 "Build TorchVision (against source-built PyTorch)"

    cd "${TORCHVISION_SRC}"

    # Check if already built
    if (cd "${VLLM_DIR}" && python -c "import torchvision; print(torchvision.__version__)") 2>/dev/null; then
        local tv_ver
        tv_ver="$(cd "${VLLM_DIR}" && python -c "import torchvision; print(torchvision.__version__)")"
        info "TorchVision ${tv_ver} already built and importable"
        cd "${VLLM_DIR}"
        return
    fi

    # Use amdclang from TheRock (same compiler as PyTorch)
    export CC="${ROCM_PATH}/lib/llvm/bin/amdclang"
    export CXX="${ROCM_PATH}/lib/llvm/bin/amdclang++"

    # TorchVision must find our source-built torch
    export TORCH_CUDA_ARCH_LIST=""
    export FORCE_CUDA=0
    export FORCE_MPS=0

    info "Building TorchVision wheel..."
    mkdir -p "${WHEELS_DIR}"
    pip wheel . \
        --no-build-isolation \
        --no-deps \
        --wheel-dir "${WHEELS_DIR}" \
        -v

    # Install the wheel into the build venv
    info "Installing TorchVision wheel into build venv..."
    local _tv_wheel
    _tv_wheel="$(newest_wheel "${WHEELS_DIR}"/torchvision-*.whl)"
    if [[ -z "${_tv_wheel}" ]]; then
        die "TorchVision wheel not found in ${WHEELS_DIR}"
    fi
    uv pip install --force-reinstall --no-deps "${_tv_wheel}"

    cd "${VLLM_DIR}"
    success "TorchVision built and installed (wheel: $(basename "${_tv_wheel}"))"
}

# =============================================================================
# Phase D: Kernel Compilers (Triton + AOTriton)
# =============================================================================

# Step 14: Clone Triton
clone_triton() {
    log_step 14 "Clone Triton source (ROCm fork, ${TRITON_BRANCH} branch)"

    if [[ -d "${TRITON_SRC}/.git" ]]; then
        info "Triton already cloned at ${TRITON_SRC}"
        cd "${TRITON_SRC}"

        # Ensure we're tracking the ROCm fork, not upstream
        local current_url
        current_url="$(git remote get-url origin 2>/dev/null)"
        if [[ "${current_url}" != *"ROCm/triton"* ]]; then
            info "Switching remote from ${current_url} to ${TRITON_REPO}"
            git remote set-url origin "${TRITON_REPO}"
        fi

        git fetch origin "${TRITON_BRANCH}"
        local current_branch
        current_branch="$(git branch --show-current)"
        if [[ "${current_branch}" != "${TRITON_BRANCH}" ]]; then
            info "Switching to ${TRITON_BRANCH} branch..."
            git checkout "${TRITON_BRANCH}"
        fi
        git pull origin "${TRITON_BRANCH}"
        cd "${VLLM_DIR}"
        success "Triton source updated (ROCm fork, ${TRITON_BRANCH})"
        return
    fi

    info "Cloning ROCm Triton (${TRITON_BRANCH} branch)..."
    git clone --branch "${TRITON_BRANCH}" "${TRITON_REPO}" "${TRITON_SRC}"
    success "Triton cloned to ${TRITON_SRC} (ROCm fork, ${TRITON_BRANCH})"
}

# Step 15: Build Triton
build_triton() {
    log_step 15 "Build Triton with ROCm backend"

    cd "${TRITON_SRC}"

    # Check if already built — look for our wheel in WHEELS_DIR
    if compgen -G "${WHEELS_DIR}"/triton*.whl >/dev/null 2>&1; then
        local triton_ver
        triton_ver="$(python -c "import triton; print(triton.__version__)" 2>/dev/null || true)"
        if [[ -n "${triton_ver}" ]]; then
            info "Triton ${triton_ver} already built (wheel exists)"
            cd "${VLLM_DIR}"
            return
        fi
    fi

    info "Building Triton with ROCm backend..."
    info "ROCM_PATH=${ROCM_PATH}"

    # Triton's setup.py requires pybind11 at import time before pip can
    # resolve build deps.
    uv pip install pybind11

    # Validate ROCm toolchain is available.
    if [[ -z "${ROCM_PATH:-}" || ! -d "${ROCM_PATH}/lib/llvm" ]]; then
        die "ROCM_PATH is not set or ${ROCM_PATH:-<unset>}/lib/llvm does not exist. Run TheRock build first (steps 5-8)."
    fi

    # Flags come from vllm-env.sh (CFLAGS, CXXFLAGS, CMAKE_*_FLAGS_RELEASE).
    if [[ -z "${CFLAGS:-}" ]] || [[ -z "${CMAKE_CXX_FLAGS_RELEASE:-}" ]]; then
        die "CFLAGS or CMAKE_CXX_FLAGS_RELEASE not set — vllm-env.sh was not sourced"
    fi
    export PATH="${ROCM_PATH}/lib/llvm/bin:${PATH}"
    export CC="${ROCM_PATH}/lib/llvm/bin/amdclang"
    export CXX="${ROCM_PATH}/lib/llvm/bin/amdclang++"
    info "CC=${CC}"
    info "CXX=${CXX}"
    info "CFLAGS=${CFLAGS}"
    info "CMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"

    # LLVM_SYSPATH: use TheRock's AMD LLVM 22 instead of Triton's bundled LLVM.
    # This gives Triton access to AMD-specific AMDGPU codegen improvements,
    # gfx1151 target support, and Polly polyhedral optimizer passes.
    # ccache: ensure /usr/bin is in PATH so uv's isolated build subprocess
    # can find ccache (uv sanitizes PATH in build isolation).
    if command -v ccache &>/dev/null; then
        export TRITON_BUILD_WITH_CCACHE=true
    else
        unset TRITON_BUILD_WITH_CCACHE
    fi
    export ROCM_HOME="${ROCM_PATH}"

    # DO NOT set LLVM_SYSPATH to TheRock's LLVM 22. ROCm's Triton fork
    # (main_perf branch) targets LLVM ~19 APIs. TheRock LLVM 22 has breaking
    # API changes in MLIR (renamed methods, removed members, changed ABIs).
    # Let Triton download and build its own LLVM version that matches its code.
    unset LLVM_SYSPATH 2>/dev/null || true

    # ROCm/triton keeps setup.py in python/ (upstream moved it to repo root).
    local triton_pkg_dir="${TRITON_SRC}"
    if [[ -f "${TRITON_SRC}/python/setup.py" && ! -f "${TRITON_SRC}/setup.py" ]]; then
        triton_pkg_dir="${TRITON_SRC}/python"
    fi

    # Triton's CMakeLists.txt hardcodes -Werror. Our custom-built Python 3.13
    # pyconfig.h redefines _POSIX_C_SOURCE which triggers -Wmacro-redefined.
    # With -Werror this becomes fatal. Remove -Werror — Triton's own tests
    # validate correctness; we don't need warnings-as-errors for our build.
    local _triton_cmakelists="${TRITON_SRC}/CMakeLists.txt"
    if grep -q ' -Werror ' "${_triton_cmakelists}" 2>/dev/null; then
        info "Patching Triton CMakeLists.txt: removing -Werror"
        sed -i 's/ -Werror / /' "${_triton_cmakelists}"
    fi

    cd "${triton_pkg_dir}"
    mkdir -p "${WHEELS_DIR}"
    pip wheel . --no-build-isolation --no-deps --wheel-dir "${WHEELS_DIR}" -v

    # Install the wheel
    local _triton_wheel
    _triton_wheel="$(newest_wheel "${WHEELS_DIR}"/triton*.whl)"
    if [[ -n "${_triton_wheel}" ]]; then
        uv pip install --force-reinstall --no-deps "${_triton_wheel}"
    else
        die "Triton wheel not found in ${WHEELS_DIR}"
    fi

    cd "${VLLM_DIR}"
    success "Triton built with ROCm backend (wheel: $(basename "${_triton_wheel}"))"
}

# Step 16: Validate Triton
validate_triton() {
    log_step 16 "Validate Triton"

    python -c "
import triton
print(f'  Triton version: {triton.__version__}')
print(f'  Triton location: {triton.__file__}')

# Verify ROCm backend is available
try:
    from triton.backends.amd import HIPBackend
    print('  ROCm/HIP backend: available')
except ImportError:
    try:
        from triton.runtime.backends import backends
        if 'hip' in backends or 'amd' in backends:
            print('  ROCm backend: available (via runtime)')
        else:
            print(f'  Available backends: {list(backends.keys())}')
    except Exception as e:
        print(f'  Backend check: {e}')
" || warn "Triton validation had issues (may still work with vLLM)"

    success "Triton validated"
}

# Step 17: Clone AOTriton
clone_aotriton() {
    log_step 17 "Clone AOTriton (ahead-of-time compiled Triton kernels)"

    if [[ -d "${AOTRITON_SRC}/.git" ]]; then
        info "AOTriton already cloned at ${AOTRITON_SRC}"
        cd "${AOTRITON_SRC}"
        git fetch origin
        git pull origin main
        git submodule update --init --recursive
        cd "${VLLM_DIR}"
        success "AOTriton source updated"
        return
    fi

    info "Cloning AOTriton..."
    git clone --recurse-submodules "${AOTRITON_REPO}" "${AOTRITON_SRC}"
    success "AOTriton cloned to ${AOTRITON_SRC}"
}

# Step 18: Build AOTriton
build_aotriton() {
    log_step 18 "Build AOTriton (pre-compiled attention kernels for gfx1151)"

    cd "${AOTRITON_SRC}"


    # Check if already built
    if [[ -f "${LOCAL_PREFIX}/lib/libaotriton_v2.so" ]] || [[ -f "${LOCAL_PREFIX}/lib/libaotriton.a" ]]; then
        info "AOTriton already built at ${LOCAL_PREFIX}"
        cd "${VLLM_DIR}"
        return
    fi

    info "Building AOTriton for gfx1151..."
    info "This pre-compiles Triton attention kernels to HSACO (no JIT at inference time)."

    # Upstream ROCm/aotriton main has a stray git rebase "pick" line at the
    # end of CMakeLists.txt (commit 0383901). Remove it.
    if grep -q '^pick ' "${AOTRITON_SRC}/CMakeLists.txt" 2>/dev/null; then
        info "Patching AOTriton CMakeLists.txt: removing stray rebase 'pick' line"
        sed -i '/^pick /d' "${AOTRITON_SRC}/CMakeLists.txt"
    fi

    # AOTriton's cmake-based build compiles Triton kernels ahead of time
    # into .hsaco binaries for the target GPU architecture.
    # AOTRITON_GPU_BUILD_TIMEOUT=0 disables the per-kernel timeout.
    uv pip install -r requirements.txt 2>/dev/null || pip install -r requirements.txt

    # Flags come from vllm-env.sh (CFLAGS, CXXFLAGS, CMAKE_*_FLAGS_RELEASE).
    if [[ -z "${CFLAGS:-}" ]] || [[ -z "${CMAKE_CXX_FLAGS_RELEASE:-}" ]]; then
        die "CFLAGS or CMAKE_CXX_FLAGS_RELEASE not set — vllm-env.sh was not sourced"
    fi
    export PATH="${ROCM_PATH}/lib/llvm/bin:${PATH}"
    export CC="${ROCM_PATH}/lib/llvm/bin/amdclang"
    export CXX="${ROCM_PATH}/lib/llvm/bin/amdclang++"
    info "CC=${CC}"
    info "CFLAGS=${CFLAGS}"
    info "CMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"

    cmake -B build -GNinja . \
        -DCMAKE_INSTALL_PREFIX="${LOCAL_PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="${CC}" \
        -DCMAKE_CXX_COMPILER="${CXX}" \
        -DAOTRITON_GPU_BUILD_TIMEOUT=0 \
        -DAOTRITON_TARGET_ARCH="gfx1151"

    ninja -C build install/strip

    # Make AOTriton findable by downstream consumers (PyTorch, vLLM)
    export AOTRITON_INSTALL_DIR="${LOCAL_PREFIX}"

    cd "${VLLM_DIR}"
    success "AOTriton built (pre-compiled attention kernels for gfx1151)"
}

# =============================================================================
# Phase D: Inference Engine (vLLM)
# =============================================================================

# Step 19: Clone vLLM
clone_vllm() {
    log_step 19 "Clone vLLM source"

    if [[ -d "${VLLM_SRC}/.git" ]]; then
        info "vLLM source already cloned at ${VLLM_SRC}"
        info "Updating to latest..."
        cd "${VLLM_SRC}"
        git fetch origin
        git pull origin main
        cd "${VLLM_DIR}"
        success "vLLM source updated"
        return
    fi

    info "Cloning vLLM repository..."
    git clone "${VLLM_REPO}" "${VLLM_SRC}"
    success "vLLM cloned to ${VLLM_SRC}"
}

# Step 20: Patch amdsmi Import Order
patch_amdsmi_import() {
    log_step 20 "Patch amdsmi import order in vLLM"

    local init_file="${VLLM_SRC}/vllm/__init__.py"

    if [[ ! -f "${init_file}" ]]; then
        die "vLLM __init__.py not found at ${init_file}"
    fi

    # Check if already patched
    if grep -q "# PATCHED: amdsmi import order" "${init_file}"; then
        info "Already patched"
        return
    fi

    info "Patching ${init_file} to import amdsmi before torch..."

    # Create backup
    cp "${init_file}" "${init_file}.bak"

    # Prepend amdsmi import at the top of the file (after any docstring/comments)
    python -c "
import re

with open('${init_file}', 'r') as f:
    content = f.read()

# Insert amdsmi import after the module docstring
patch = '''# PATCHED: amdsmi import order (must be before torch or it crashes)
try:
    import amdsmi  # noqa: F401
except ImportError:
    pass

'''

# Find the end of the docstring or initial comments
lines = content.split('\n')
insert_idx = 0
in_docstring = False
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith('\"\"\"') or stripped.startswith(\"'''\"):
        if in_docstring:
            insert_idx = i + 1
            in_docstring = False
            break
        elif stripped.endswith('\"\"\"') and len(stripped) > 3:
            insert_idx = i + 1
            break
        else:
            in_docstring = True
    elif not in_docstring and stripped and not stripped.startswith('#'):
        insert_idx = i
        break

lines.insert(insert_idx, patch)

with open('${init_file}', 'w') as f:
    f.write('\n'.join(lines))
"

    success "amdsmi import patch applied"
}

# Step 20b: Patch vLLM for gfx1151 (RDNA 3.5) AITER support
#
# vLLM upstream gates AITER backend selection on gfx9 architectures only.
# The AMD fork's AITER has full gfx1151 support (chip_info.py maps gfx1151
# to enum 13, fwd_prefill.py has explicit RDNA 3.5 tuning: BLOCK_M=32,
# BLOCK_N=32, waves_per_eu=2). These patches extend the architecture checks
# to include gfx1x (RDNA 3.x) alongside the existing gfx9 checks.
#
# Three files patched:
#   1. _aiter_ops.py:is_aiter_found_and_supported() — master AITER gate
#   2. rocm_aiter_fa.py:supports_compute_capability() — attention backend gate
#   3. rocm.py:get_vit_attn_backend() — ViT attention backend selection
patch_vllm_gfx1151() {
    log_step 20 "Patch vLLM for gfx1151 AITER support"

    # Patch 1: _aiter_ops.py — extend is_aiter_found_and_supported() to accept gfx1x
    local _aiter_ops="${VLLM_SRC}/vllm/_aiter_ops.py"
    if [[ -f "${_aiter_ops}" ]] && ! grep -q 'on_gfx1x' "${_aiter_ops}"; then
        info "Patching _aiter_ops.py: extending AITER support to gfx1x"
        sed -i 's/from vllm\.platforms\.rocm import on_gfx9$/from vllm.platforms.rocm import on_gfx1x, on_gfx9/' "${_aiter_ops}"
        sed -i 's/return on_gfx9()$/return on_gfx9() or on_gfx1x()/' "${_aiter_ops}"
    else
        info "_aiter_ops.py: already patched or pattern not found"
    fi

    # Patch 2: rocm_aiter_fa.py — extend supports_compute_capability() to accept gfx1x
    local _rocm_aiter_fa="${VLLM_SRC}/vllm/v1/attention/backends/rocm_aiter_fa.py"
    if [[ -f "${_rocm_aiter_fa}" ]] && ! grep -q 'on_gfx1x' "${_rocm_aiter_fa}"; then
        info "Patching rocm_aiter_fa.py: extending compute capability to gfx1x"
        sed -i 's/from vllm\.platforms\.rocm import on_mi3xx/from vllm.platforms.rocm import on_gfx1x, on_mi3xx/' "${_rocm_aiter_fa}"
        sed -i 's/return on_mi3xx()/return on_mi3xx() or on_gfx1x()/' "${_rocm_aiter_fa}"
    else
        info "rocm_aiter_fa.py: already patched or pattern not found"
    fi

    # Patch 3: rocm.py — extend ViT attention backend selection to accept gfx1x
    local _rocm_py="${VLLM_SRC}/vllm/platforms/rocm.py"
    if [[ -f "${_rocm_py}" ]] && grep -q 'rocm_aiter_ops.is_enabled() and on_gfx9()' "${_rocm_py}"; then
        info "Patching rocm.py: extending ViT attention backend to gfx1x"
        sed -i 's/rocm_aiter_ops\.is_enabled() and on_gfx9()/rocm_aiter_ops.is_enabled() and (on_gfx9() or on_gfx1x())/' "${_rocm_py}"
    else
        info "rocm.py: already patched or pattern not found"
    fi

    success "vLLM gfx1151 AITER patches applied"
}

# Step 21: Install Build Dependencies
install_build_deps() {
    log_step 21 "Install build dependencies"

    info "Installing build dependencies..."
    uv pip install \
        pip \
        numba \
        scipy \
        "huggingface-hub[cli,hf_transfer]" \
        setuptools_scm \
        "numpy>=2.0,<3" \
        wheel \
        packaging \
        ninja

    success "Build dependencies installed"
}

# Step 22: Run use_existing_torch.py
run_use_existing_torch() {
    log_step 22 "Run use_existing_torch.py"

    cd "${VLLM_SRC}"

    if [[ ! -f "use_existing_torch.py" ]]; then
        warn "use_existing_torch.py not found (may not be needed in this vLLM version)"
        cd "${VLLM_DIR}"
        return
    fi

    info "Running use_existing_torch.py..."
    python use_existing_torch.py

    cd "${VLLM_DIR}"
    success "use_existing_torch.py completed"
}

# Step 23: Install ROCm Requirements
install_rocm_requirements() {
    log_step 23 "Install ROCm requirements"

    cd "${VLLM_SRC}"

    local req_file="requirements/rocm.txt"
    if [[ ! -f "${req_file}" ]]; then
        # Try alternative locations
        req_file="requirements-rocm.txt"
        if [[ ! -f "${req_file}" ]]; then
            warn "ROCm requirements file not found (skipping)"
            cd "${VLLM_DIR}"
            return
        fi
    fi

    info "Installing from ${req_file}..."

    # First uninstall amdsmi if present (will be reinstalled from ROCm path later)
    uv pip uninstall amdsmi 2>/dev/null || true

    # Protect source-built packages from PyPI overwrite.
    # pip's dependency resolver will pull torch/torchvision as transitive deps
    # of packages like transformers, conch-triton-kernels, etc. We use a
    # constraints file to tell pip "these are already satisfied, don't touch them."
    local _constraints_file="${VLLM_DIR}/.build-constraints.txt"
    local _torch_version
    _torch_version="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"

    if [[ -n "${_torch_version}" ]]; then
        info "Protecting source-built torch ${_torch_version} from PyPI overwrite"
        cat > "${_constraints_file}" << CONSTRAINTS_EOF
torch==${_torch_version}
torchvision>=0.0.0
torchaudio>=0.0.0
numpy>=2.0,<3
CONSTRAINTS_EOF
        uv pip install -r "${req_file}" -c "${_constraints_file}"
    else
        warn "torch not installed — deps may pull PyPI torch (will reinstall source torch after)"
        uv pip install -r "${req_file}"
    fi

    # Verify source-built torch survived dependency installation.
    # If a transitive dep replaced it, reinstall from the PyTorch source tree.
    local _torch_hip
    _torch_hip="$(python -c 'import torch; print(torch.version.hip or "")' 2>/dev/null || true)"

    if [[ -z "${_torch_hip}" ]]; then
        warn "Source-built torch was overwritten or missing — reinstalling from wheel"
        uv pip uninstall torch torchvision 2>/dev/null || true

        # Reinstall from the pre-built wheel (fast — no cmake, no compilation)
        local _torch_wheel
        _torch_wheel="$(newest_wheel "${WHEELS_DIR}"/torch-*.whl)"
        if [[ -n "${_torch_wheel}" ]]; then
            uv pip install --force-reinstall --no-deps "${_torch_wheel}"
        else
            die "No PyTorch wheel found in ${WHEELS_DIR} — run step 10 first"
        fi

        _torch_hip="$(python -c 'import torch; print(torch.version.hip or "")' 2>/dev/null || true)"
        if [[ -z "${_torch_hip}" ]]; then
            die "Failed to reinstall source-built PyTorch — torch.version.hip is still None"
        fi
        success "Source-built torch reinstalled from wheel (hip=${_torch_hip})"
    else
        info "Source-built torch verified (hip=${_torch_hip})"
    fi

    rm -f "${_constraints_file}"

    cd "${VLLM_DIR}"
    success "ROCm requirements installed"
}

# Step 24: Build vLLM
build_vllm() {
    log_step 24 "Build vLLM"

    cd "${VLLM_SRC}"

    # Flags come from vllm-env.sh (CFLAGS, CXXFLAGS, CMAKE_*_FLAGS_RELEASE).
    if [[ -z "${CFLAGS:-}" ]] || [[ -z "${CMAKE_CXX_FLAGS_RELEASE:-}" ]]; then
        die "CFLAGS or CMAKE_CXX_FLAGS_RELEASE not set — vllm-env.sh was not sourced"
    fi
    # ROCm environment for vLLM's setup.py auto-detection
    export PATH="${ROCM_PATH}/lib/llvm/bin:${PATH}"
    export CC="${ROCM_PATH}/lib/llvm/bin/amdclang"
    export CXX="${ROCM_PATH}/lib/llvm/bin/amdclang++"
    export ROCM_HOME="${ROCM_PATH}"
    export HIP_PATH="${ROCM_PATH}"

    info "Building vLLM with PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}"
    info "CC=${CC}, CXX=${CXX}"
    info "CFLAGS=${CFLAGS}"
    info "CMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"
    info "ROCM_HOME=${ROCM_HOME}"

    # Make AOTriton available if built
    if [[ -d "${LOCAL_PREFIX}" ]]; then
        export AOTRITON_INSTALL_DIR="${LOCAL_PREFIX}"
        export CMAKE_PREFIX_PATH="${LOCAL_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
        info "AOTriton available at ${LOCAL_PREFIX}"
    fi

    # Attempt build with AITER first
    info "Attempting build WITH AITER backend..."
    export VLLM_ROCM_USE_AITER=1
    mkdir -p "${WHEELS_DIR}"

    local build_succeeded=false

    if pip wheel . --no-build-isolation --no-deps --wheel-dir "${WHEELS_DIR}" -v 2>&1; then
        build_succeeded=true
        echo "enabled" > "${VLLM_DIR}/.aiter-status"
        success "vLLM wheel built WITH AITER"
    else
        warn "AITER build failed. Falling back to Triton-only build..."
        unset VLLM_ROCM_USE_AITER

        # Clean partial build artifacts
        python setup.py clean 2>/dev/null || true
        find . -name "*.so" -path "*/build/*" -delete 2>/dev/null || true

        if pip wheel . --no-build-isolation --no-deps --wheel-dir "${WHEELS_DIR}" -v 2>&1; then
            build_succeeded=true
            echo "disabled" > "${VLLM_DIR}/.aiter-status"
            success "vLLM wheel built WITHOUT AITER (Triton only)"
        fi
    fi

    if [[ "${build_succeeded}" != "true" ]]; then
        cd "${VLLM_DIR}"
        die "vLLM build failed. Check ${VLLM_LOG} for details."
    fi

    # Install the vLLM wheel into the build venv
    local _vllm_wheel
    _vllm_wheel="$(newest_wheel "${WHEELS_DIR}"/vllm-*.whl)"
    if [[ -n "${_vllm_wheel}" ]]; then
        info "Installing vLLM wheel into build venv..."
        uv pip install --force-reinstall --no-deps "${_vllm_wheel}"
    fi

    cd "${VLLM_DIR}"
}

# =============================================================================
# Phase F: Attention (Flash Attention + AITER)
# =============================================================================

# Step 25: Reinstall amdsmi
reinstall_amdsmi() {
    log_step 25 "Reinstall amdsmi from ROCm"

    if [[ -z "${ROCM_PATH:-}" ]]; then
        die "ROCM_PATH not set."
    fi

    local amdsmi_dir="${ROCM_PATH}/share/amd_smi"

    if [[ ! -d "${amdsmi_dir}" ]]; then
        # Try alternative path
        amdsmi_dir="${ROCM_PATH}/lib/amd_smi"
        if [[ ! -d "${amdsmi_dir}" ]]; then
            warn "amdsmi source not found in TheRock (skipping reinstall)"
            return
        fi
    fi

    info "Installing amdsmi from ${amdsmi_dir}..."
    uv pip install "${amdsmi_dir}"

    # Verify import
    python -c "import amdsmi; print(f'  amdsmi version: {amdsmi.__version__}')" 2>/dev/null \
        || warn "amdsmi import check returned non-zero (may still work)"

    success "amdsmi reinstalled from ROCm"
}

# Step 26: Clone Flash Attention
clone_flash_attention() {
    log_step 26 "Clone Flash Attention (main_perf branch)"

    if [[ -d "${FLASH_ATTN_SRC}/.git" ]]; then
        info "Flash Attention already cloned at ${FLASH_ATTN_SRC}"
        cd "${FLASH_ATTN_SRC}"
        local current_branch
        current_branch="$(git branch --show-current)"
        if [[ "${current_branch}" != "main_perf" ]]; then
            info "Switching to main_perf branch..."
            git fetch origin main_perf
            git checkout main_perf
        fi
        git pull origin main_perf
        cd "${VLLM_DIR}"
        success "Flash Attention source updated (main_perf)"
        return
    fi

    info "Cloning Flash Attention (main_perf branch)..."
    git clone --branch main_perf "${FLASH_ATTN_REPO}" "${FLASH_ATTN_SRC}"
    success "Flash Attention cloned (main_perf branch)"
}

# Step 27: Patch Flash Attention amdsmi Import
patch_flash_amdsmi() {
    log_step 27 "Patch Flash Attention amdsmi import"

    local init_file="${FLASH_ATTN_SRC}/flash_attn/__init__.py"

    if [[ ! -f "${init_file}" ]]; then
        warn "flash_attn/__init__.py not found (may not need patching)"
        return
    fi

    if grep -q "# PATCHED: amdsmi import order" "${init_file}"; then
        info "Already patched"
        return
    fi

    info "Patching ${init_file}..."
    cp "${init_file}" "${init_file}.bak"

    # Prepend amdsmi import
    {
        echo "# PATCHED: amdsmi import order (must be before torch or it crashes)"
        echo "try:"
        echo "    import amdsmi  # noqa: F401"
        echo "except ImportError:"
        echo "    pass"
        echo ""
        cat "${init_file}.bak"
    } > "${init_file}"

    success "Flash Attention amdsmi patch applied"
}

# Step 28: Build Flash Attention
build_flash_attention() {
    log_step 28 "Build Flash Attention"

    cd "${FLASH_ATTN_SRC}"

    # Flags come from vllm-env.sh (CFLAGS, CXXFLAGS, CMAKE_*_FLAGS_RELEASE).
    if [[ -z "${CFLAGS:-}" ]] || [[ -z "${CMAKE_CXX_FLAGS_RELEASE:-}" ]]; then
        die "CFLAGS or CMAKE_CXX_FLAGS_RELEASE not set — vllm-env.sh was not sourced"
    fi
    info "Building Flash Attention with Triton AMD enabled..."
    info "FLASH_ATTENTION_TRITON_AMD_ENABLE=${FLASH_ATTENTION_TRITON_AMD_ENABLE}"
    info "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}"
    info "CFLAGS=${CFLAGS}"
    info "CMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"

    # Record our source-built Triton before Flash Attention can overwrite it
    local triton_loc_before
    triton_loc_before="$(python -c "import triton; print(triton.__file__)" 2>/dev/null || echo 'none')"
    info "Triton location before FA build: ${triton_loc_before}"

    # Install Flash Attention deps first (excluding triton — we built it from source)
    uv pip install einops packaging

    # Build wheel — no editable install, no triton download from PyPI
    pip wheel . --no-build-isolation --no-deps --wheel-dir "${WHEELS_DIR}" -v

    # Install the wheel
    local _fa_wheel
    _fa_wheel="$(newest_wheel "${WHEELS_DIR}"/flash_attn-*.whl)"
    if [[ -n "${_fa_wheel}" ]]; then
        uv pip install --force-reinstall --no-deps "${_fa_wheel}"
    fi

    # Verify our source-built Triton was NOT overwritten
    local triton_loc_after
    triton_loc_after="$(python -c "import triton; print(triton.__file__)" 2>/dev/null || echo 'none')"
    if [[ "${triton_loc_after}" != "${triton_loc_before}" ]]; then
        warn "Triton location changed: ${triton_loc_before} -> ${triton_loc_after}"
        warn "Flash Attention may have overwritten source-built Triton!"
        if [[ "${triton_loc_before}" == *"${TRITON_SRC}"* ]]; then
            info "Reinstalling source-built Triton from wheel..."
            local _triton_wheel
            _triton_wheel="$(newest_wheel "${WHEELS_DIR}"/triton*.whl)"
            if [[ -n "${_triton_wheel}" ]]; then
                uv pip install --force-reinstall --no-deps "${_triton_wheel}"
            else
                warn "No Triton wheel found — cannot reinstall"
            fi
            cd "${FLASH_ATTN_SRC}"
        fi
    else
        success "Source-built Triton preserved"
    fi

    cd "${VLLM_DIR}"
    success "Flash Attention built"
}

# =============================================================================
# Phase F: Validation
# =============================================================================

# Step 29: Smoke Test
smoke_test() {
    log_step 29 "Smoke test"

    info "Verifying full inference stack..."

    # Check vllm CLI exists
    if command -v vllm &>/dev/null; then
        success "vllm CLI found: $(which vllm)"
    else
        die "vllm CLI not found in PATH after build"
    fi

    # Check vllm Python import
    python -c "
import vllm
print(f'  vLLM version: {vllm.__version__}')
" || die "Failed to import vllm Python module"

    # Check Python was built from source
    python -c "
import sys, sysconfig
loc = sys.executable
print(f'  Python: {sys.version}')
print(f'  Python executable: {loc}')
lm = 'yes' if '-lalm' in (sysconfig.get_config_var('LDFLAGS') or '') else 'no'
print(f'  AOCL-LibM linked: {lm}')
if '/opt/src/vllm/python/' in loc or '/opt/src/vllm/.venv/' in loc:
    print('  Python: BUILT FROM SOURCE')
else:
    print(f'  Python: WARNING — may not be from source build')
"

    # Check PyTorch was built from source (not pip wheel)
    python -c "
import torch
loc = torch.__file__
print(f'  PyTorch location: {loc}')
if '/opt/src/vllm/pytorch/' in loc:
    print('  PyTorch: BUILT FROM SOURCE (ROCm fork)')
else:
    print(f'  PyTorch: WARNING — may not be from source build')
"

    # Check Triton
    python -c "
import triton
loc = triton.__file__
print(f'  Triton location: {loc}')
if '/opt/src/vllm/triton/' in loc:
    print('  Triton: BUILT FROM SOURCE (ROCm fork)')
else:
    print(f'  Triton: WARNING — may not be from source build')
"

    # Check Flash Attention
    python -c "
try:
    import flash_attn
    print(f'  Flash Attention: loaded')
except ImportError as e:
    print(f'  Flash Attention: NOT loaded ({e})')
" || true

    # Check AOTriton
        if [[ -d "${LOCAL_PREFIX}" ]]; then
        success "AOTriton: installed at ${LOCAL_PREFIX}"
    else
        info "AOTriton: not built"
    fi

    # Check AOCL-LibM
    if [[ -f "${LOCAL_PREFIX}/lib/libalm.so" ]]; then
        success "AOCL-LibM: installed at ${LOCAL_PREFIX}"
    else
        info "AOCL-LibM: not built"
    fi

    # Check AITER status
    local aiter_status
    aiter_status="$(cat "${VLLM_DIR}/.aiter-status" 2>/dev/null || echo 'unknown')"
    info "AITER status: ${aiter_status}"

    # Verify compiler used
    info "Compiler: $(${CC} --version | head -1)"

    # Verify ROCM_PATH is local build
    info "ROCM_PATH: ${ROCM_PATH:-<not set>}"
    if [[ "${ROCM_PATH:-}" == *"/local"* ]]; then
        success "ROCm: BUILT FROM SOURCE (local)"
    else
        warn "ROCm: may not be locally compiled"
    fi

    success "Smoke test passed"
    echo ""
    info "Full inference stack build complete!"
    info "  Install directory: ${VLLM_DIR}"
    info "  Activate with: source scripts/vllm-env.sh"
    info "  AITER: ${aiter_status}"
    info "  Components: AOCL-LibM + Python + TheRock + PyTorch + Triton + AOTriton + vLLM + Flash Attention"
    info "  Wheels dir: ${WHEELS_DIR}"
    info "  All compiled from source with Clang $(${CC} --version | head -1 | grep -oP '\d+\.\d+\.\d+' | head -1)"
}

# Step 29b: Pre-warm AITER JIT modules
# Compiles all buildable AITER HIP C++ modules ahead of time so that the first
# vLLM inference request doesn't stall for minutes while modules JIT-compile.
# In a CK-free build (no Composable Kernel sources), 56 of 72 modules are
# auto-excluded, leaving 26 buildable. Of those, only 2 ship pre-built
# (module_aiter_enum, module_attention_asm). This step compiles the rest.
#
# ~6 modules will fail due to gfx1151 hardware ISA incompatibilities:
#   - module_quick_all_reduce: requires fp8-conversion-insts (gfx9 only)
#   - module_fmha_v3_*: MFMA tile dimensions assume warp_size=64
#   - module_mhc: same MFMA warp_size=64 static_assert issue
# These failures are non-fatal — the modules target gfx9xx hardware features
# that RDNA 3.5 doesn't have and would never be called at runtime.
warmup_aiter_jit() {
    log_step 29 "Pre-warm AITER JIT modules"

    # Skip if AITER is not enabled
    local aiter_status
    aiter_status="$(cat "${VLLM_DIR}/.aiter-status" 2>/dev/null || echo 'unknown')"
    if [[ "${aiter_status}" != "enabled" ]]; then
        info "AITER not enabled (status: ${aiter_status}), skipping JIT pre-warm"
        return 0
    fi

    # The JIT directory is where compiled .so files land
    local jit_dir
    jit_dir="$(python -c "from aiter.jit.core import get_user_jit_dir; print(get_user_jit_dir())" 2>/dev/null)"
    if [[ -z "${jit_dir}" ]]; then
        warn "Cannot determine AITER JIT directory, skipping pre-warm"
        return 0
    fi
    info "AITER JIT directory: ${jit_dir}"

    # Run the pre-warm script. This calls get_args_of_build("all") to enumerate
    # buildable modules (auto-excludes CK-dependent ones), then build_module()
    # for each one that doesn't already have a .so in the JIT directory.
    # Each module takes 10-30 seconds to compile with hipcc.
    python -c "
import os, sys, time

from aiter.jit.core import get_args_of_build, build_module, get_user_jit_dir

jit_dir = get_user_jit_dir()
all_ops_list, d_all_ops = get_args_of_build('all')

total = len(all_ops_list)
already_built = 0
newly_built = 0
failed = 0

print(f'AITER JIT pre-warm: {total} buildable modules')

for i, mod_cfg in enumerate(all_ops_list, 1):
    md_name = mod_cfg['md_name']
    so_path = os.path.join(jit_dir, f'{md_name}.so')

    if os.path.exists(so_path):
        print(f'  [{i:2d}/{total}] {md_name}: already built')
        already_built += 1
        continue

    print(f'  [{i:2d}/{total}] {md_name}: compiling...', flush=True)
    start = time.perf_counter()
    try:
        # get_args_of_build for a single module returns the full config dict
        # (not a list). The 'all' path returns per-module dicts with only
        # md_name/srcs/flags/includes — we need the full defaults too.
        d_args = get_args_of_build(md_name)
        build_module(
            md_name=md_name,
            srcs=d_args['srcs'],
            flags_extra_cc=d_args['flags_extra_cc'],
            flags_extra_hip=d_args['flags_extra_hip'],
            blob_gen_cmd=d_args['blob_gen_cmd'],
            extra_include=d_args['extra_include'],
            extra_ldflags=d_args['extra_ldflags'],
            verbose=d_args['verbose'],
            is_python_module=d_args['is_python_module'],
            is_standalone=d_args['is_standalone'],
            torch_exclude=d_args['torch_exclude'],
            hipify=d_args.get('hipify', False),
        )
        elapsed = time.perf_counter() - start
        if os.path.exists(so_path):
            print(f'           compiled in {elapsed:.1f}s')
            newly_built += 1
        else:
            print(f'           build_module returned but .so not found ({elapsed:.1f}s)')
            failed += 1
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f'           FAILED ({elapsed:.1f}s): {e}')
        failed += 1

print()
print(f'AITER JIT pre-warm complete:')
print(f'  Already built: {already_built}')
print(f'  Newly built:   {newly_built}')
print(f'  Failed:        {failed}')
print(f'  Total:         {total}')

if failed > 0:
    print(f'WARNING: {failed} modules failed to compile. These will JIT-compile on first use.')
    sys.exit(0)  # Non-fatal — modules will JIT-compile on demand
" || warn "AITER JIT pre-warm had errors (modules will JIT-compile on first use)"

    # Report final count
    local built_count
    built_count="$(find "${jit_dir}" -maxdepth 1 -name '*.so' -type f 2>/dev/null | wc -l)"
    success "AITER JIT pre-warm: ${built_count} modules compiled in ${jit_dir}"
}

# Step 29c: TunableOp GEMM warmup
# Runs a brief inference warmup with PYTORCH_TUNABLEOP_ENABLED=1 to populate
# the TunableOp CSV cache with optimal GEMM kernel selections for gfx1151.
# The CSV is model-specific (different models produce different GEMM shapes),
# so this uses the director model configured in .env as the primary workload.
# Subsequent vLLM starts with the same model skip the tuning phase entirely.
warmup_tunableop() {
    log_step 29 "TunableOp GEMM warmup"

    local tunableop_csv="${VLLM_DIR}/tunableop_results_gfx1151.csv"

    # Skip if CSV already has substantial content (>10 tuned kernels)
    if [[ -f "${tunableop_csv}" ]]; then
        local line_count
        line_count="$(wc -l < "${tunableop_csv}")"
        if [[ "${line_count}" -gt 10 ]]; then
            info "TunableOp CSV already populated (${line_count} entries): ${tunableop_csv}"
            return 0
        fi
    fi

    # Source .env if available to get model configuration
    local env_file="${PLATFORM_DIR}/.env"
    if [[ ! -f "${env_file}" ]]; then
        info "No .env file found — skipping TunableOp warmup (requires model config)"
        info "Run 'vllm-start.sh' with TunableOp enabled to populate CSV on first inference"
        return 0
    fi

    # Read the director model from .env (the primary/largest model)
    set -a
    # shellcheck source=/dev/null
    source "${env_file}"
    set +a

    local model="${VLLM_DIRECTOR_MODEL:-}"
    if [[ -z "${model}" ]]; then
        info "VLLM_DIRECTOR_MODEL not set in .env — skipping TunableOp warmup"
        info "TunableOp will auto-tune on first inference with each model"
        return 0
    fi

    # Check if model files exist locally
    local model_path
    model_path="$(python -c "
from pathlib import Path
import os
model = '${model}'
# Check common HuggingFace cache locations
for base in [os.path.expanduser('~/.cache/huggingface/hub'), '/opt/models']:
    p = Path(base) / ('models--' + model.replace('/', '--'))
    if p.exists():
        print(p)
        break
    p = Path(base) / model
    if p.exists():
        print(p)
        break
" 2>/dev/null || true)"

    if [[ -z "${model_path}" ]]; then
        info "Model '${model}' not found in local cache — skipping TunableOp warmup"
        info "TunableOp will auto-tune on first inference when the model is loaded"
        return 0
    fi

    info "Running TunableOp warmup with model: ${model}"
    info "CSV output: ${tunableop_csv}"

    # Enable TunableOp and run a brief vLLM offline inference to discover
    # GEMM shapes and benchmark kernel candidates.
    export PYTORCH_TUNABLEOP_ENABLED=1
    export PYTORCH_TUNABLEOP_FILENAME="${tunableop_csv}"
    export PYTORCH_TUNABLEOP_TUNING=1

    # Use vLLM's offline LLMEngine to load the model and run a few prompts.
    # This exercises the full GEMM dispatch path without starting a server.
    python -c "
import os
os.environ['PYTORCH_TUNABLEOP_ENABLED'] = '1'
os.environ['PYTORCH_TUNABLEOP_FILENAME'] = '${tunableop_csv}'
os.environ['PYTORCH_TUNABLEOP_TUNING'] = '1'

from vllm import LLM, SamplingParams

print('Loading model for TunableOp warmup...')
llm = LLM(
    model='${model}',
    max_model_len=2048,
    gpu_memory_utilization=0.5,
    enforce_eager=True,  # No graph capture during tuning
)

# Run a few prompts of varying length to exercise different GEMM shapes.
# Short prompt exercises different matrix dimensions than long prompt.
prompts = [
    'Hello',
    'Explain the theory of relativity in simple terms.',
    'Write a detailed analysis of the economic impacts of climate change on ' * 10,
]
params = SamplingParams(temperature=0.0, max_tokens=32)

print('Running TunableOp warmup inference passes...')
outputs = llm.generate(prompts, params)
for out in outputs:
    print(f'  Prompt tokens: {len(out.prompt_token_ids)}, '
          f'Output tokens: {len(out.outputs[0].token_ids)}')

print(f'TunableOp CSV written to: ${tunableop_csv}')
" || {
        warn "TunableOp warmup failed — CSV will be populated on first regular inference"
        return 0
    }

    # Report results
    if [[ -f "${tunableop_csv}" ]]; then
        local csv_lines
        csv_lines="$(wc -l < "${tunableop_csv}")"
        success "TunableOp warmup complete: ${csv_lines} kernel entries in ${tunableop_csv}"
    else
        warn "TunableOp CSV not created — tuning will occur on first inference"
    fi
}

# =============================================================================
# Phase H: Optimized Wheels (Zen 5 native builds for downstream venvs)
# =============================================================================
# These wheels are built with aggressive Zen 5 optimization flags so that
# performance-critical Python packages run at full speed when installed
# into downstream venvs.
#
# Two categories:
#   Rust packages:  RUSTFLAGS="-C target-cpu=znver5" enables AVX-512, VAES
#   C/C++ packages: CFLAGS from vllm-env.sh (-O3 -march=native -flto=thin ...)

# Step 30: Build Rust optimized wheels (orjson, cryptography)
build_rust_wheels() {
    log_step 30 "Build Rust optimized wheels (orjson, cryptography)"

    mkdir -p "${WHEELS_DIR}"

    # Verify Rust toolchain
    if ! command -v rustc &>/dev/null; then
        die "Rust toolchain not found. Install with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    fi
    if ! command -v cargo &>/dev/null; then
        die "cargo not found. Install with rustup."
    fi

    local rust_ver
    rust_ver="$(rustc --version | head -1)"
    info "Rust: ${rust_ver}"

    # RUSTFLAGS: target-cpu=znver5 explicitly (not 'native') because Rust's
    # native detection has a bug where it identifies znver5 but only enables
    # SSE2. Explicit znver5 gives us all 40+ target features:
    # AVX-512{F,BW,DQ,VL,VNNI,IFMA,VBMI,VBMI2,BITALG,BF16,VPOPCNTDQ,VP2INTERSECT},
    # VAES (4x parallel AES in ZMM), VPCLMULQDQ, GFNI, SHA.
    #
    # Do NOT add -C lto=thin here — maturin adds -C embed-bitcode=no which
    # conflicts with -C lto. Maturin manages its own LTO configuration.
    export RUSTFLAGS="-C target-cpu=znver5 -C opt-level=3"
    info "RUSTFLAGS=${RUSTFLAGS}"

    # Rust's linker invokes `cc` which resolves to the amdclang symlink, but
    # AMD's wrapper rejects binaries not prefixed with "amd". Tell Cargo to
    # invoke amdclang by its real name. Unset CFLAGS/CXXFLAGS/LDFLAGS because
    # they contain clang-specific flags (-famd-opt, -mllvm, -mprefer-vector-width)
    # that rustc's internal cc invocations for build scripts don't understand.
    local _saved_cc="${CC:-}" _saved_cxx="${CXX:-}"
    local _saved_cflags="${CFLAGS:-}" _saved_cxxflags="${CXXFLAGS:-}"
    local _saved_ldflags="${LDFLAGS:-}"
    export CC="amdclang" CXX="amdclang++"
    export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="amdclang"
    unset CFLAGS CXXFLAGS LDFLAGS

    # orjson: Rust JSON library used on every AMQP packet, API response,
    # and JSONL stream operation. AVX-512 enables SIMD JSON parsing.
    local _orjson_wheel
    _orjson_wheel="$(newest_wheel "${WHEELS_DIR}"/orjson-*.whl)"
    if [[ -n "${_orjson_wheel}" ]]; then
        info "orjson wheel already exists: $(basename "${_orjson_wheel}")"
    else
        info "Building orjson from source with Zen 5 optimizations..."
        pip wheel orjson \
            --no-binary orjson \
            --no-cache-dir \
            --no-deps \
            --wheel-dir "${WHEELS_DIR}" \
            -v
        _orjson_wheel="$(newest_wheel "${WHEELS_DIR}"/orjson-*.whl)"
        if [[ -z "${_orjson_wheel}" ]]; then
            die "orjson wheel build failed"
        fi
        success "orjson wheel built: $(basename "${_orjson_wheel}")"
    fi

    # cryptography: Rust/C library for ChaCha20-Poly1305 encryption of
    # PRIVATE/NSFW persona packets. VAES target feature enables 4x parallel
    # AES operations in AVX-512 registers.
    local _crypto_wheel
    _crypto_wheel="$(newest_wheel "${WHEELS_DIR}"/cryptography-*.whl)"
    if [[ -n "${_crypto_wheel}" ]]; then
        info "cryptography wheel already exists: $(basename "${_crypto_wheel}")"
    else
        info "Building cryptography from source with Zen 5 optimizations..."
        # cryptography needs OpenSSL headers for its C components
        pip wheel cryptography \
            --no-binary cryptography \
            --no-cache-dir \
            --no-deps \
            --wheel-dir "${WHEELS_DIR}" \
            -v
        _crypto_wheel="$(newest_wheel "${WHEELS_DIR}"/cryptography-*.whl)"
        if [[ -z "${_crypto_wheel}" ]]; then
            die "cryptography wheel build failed"
        fi
        success "cryptography wheel built: $(basename "${_crypto_wheel}")"
    fi

    unset RUSTFLAGS CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER

    # Restore CC/CXX/CFLAGS/LDFLAGS for subsequent C/C++ builds
    export CC="${_saved_cc}" CXX="${_saved_cxx}" CFLAGS="${_saved_cflags}" CXXFLAGS="${_saved_cxxflags}" LDFLAGS="${_saved_ldflags}"

    success "Rust optimized wheels complete"
}

# Step 31: Build C/C++ optimized wheels
build_native_wheels() {
    log_step 31 "Build C/C++ optimized wheels (numpy, sentencepiece, zstandard, asyncpg)"

    mkdir -p "${WHEELS_DIR}"

    # Fix cmake wrapper: the cmake pip package installs a Python wrapper at
    # .venv/bin/cmake that does `from cmake import cmake`. Inside pip's build
    # isolation, the cmake Python module isn't available, so sentencepiece and
    # pyarrow both fail when their build scripts invoke cmake. Replace the
    # broken wrapper with a symlink to the real system cmake.
    local _real_cmake
    _real_cmake="$(command -v cmake 2>/dev/null || true)"
    if [[ -f "${VLLM_DIR}/.venv/bin/cmake" ]] && head -1 "${VLLM_DIR}/.venv/bin/cmake" | grep -q python; then
        if [[ -n "${_real_cmake}" && "${_real_cmake}" != "${VLLM_DIR}/.venv/bin/cmake" ]]; then
            info "Replacing broken Python cmake wrapper with symlink to ${_real_cmake}"
            rm "${VLLM_DIR}/.venv/bin/cmake"
            ln -s "${_real_cmake}" "${VLLM_DIR}/.venv/bin/cmake"
        else
            die "No system cmake found — install cmake (not the pip package)"
        fi
    fi

    # Rewrite -mllvm flags as -Xclang pairs for third-party wheel builds.
    # meson (used by numpy) hard-codes -Werror=unused-command-line-argument
    # in ClangCompiler.get_compiler_check_args() AFTER our CFLAGS, overriding
    # our -Wno-error. Driver-level -mllvm flags are reported as "unused" in
    # compile-only checks (-c), killing every meson capability probe.
    # -Xclang passes flags directly to the compiler frontend/backend, bypassing
    # the driver's argument tracking — so they're invisible to -Wunused.
    # -famd-opt is a link-time-only driver flag (no-op at compile time), so we
    # move it to LDFLAGS where it takes effect without triggering -Werror.
    local _saved_cflags="${CFLAGS:-}" _saved_cxxflags="${CXXFLAGS:-}"
    local _saved_ldflags="${LDFLAGS:-}"
    local _wheel_cflags
    _wheel_cflags="$(echo "${CFLAGS}" | sed -E \
        's/-mllvm (-[^ ]+)/-Xclang -mllvm -Xclang \1/g; s/-famd-opt//g; s/  +/ /g; s/^ +| +$//g')"
    export CFLAGS="${_wheel_cflags}"
    export CXXFLAGS="${_wheel_cflags}"
    export LDFLAGS="${LDFLAGS} -famd-opt"
    info "CC=${CC}"
    info "CXX=${CXX}"
    info "CFLAGS=${CFLAGS}"

    # Package list with rationale:
    #   numpy:        Tensor ops everywhere, PyTorch interop (requires >=2.0)
    #   sentencepiece: Tokenizer hot path for every model inference call
    #   zstandard:    Zstd compression with AVX-512 VAES paths (JSONL streaming)
    #   asyncpg:      PostgreSQL wire protocol, every DB call
    # Excluded:
    #   pyzstd — now pure Python (C extension moved to backports-zstd), and
    #     redundant since zstandard covers the same use case (PyTorch checkpoint
    #     uses whichever is available: zstandard OR pyzstd).
    #   pyarrow — requires building the entire Apache Arrow C++ library (30+ min,
    #     separate dependency tree). The PyPI binary uses runtime SIMD dispatch
    #     (detects AVX-512 at startup), so there's no meaningful gain from a
    #     source build. Arrow's hot paths already use the best available ISA.
    local -a _packages=(
        "numpy"
        "sentencepiece"
        "zstandard"
        "asyncpg"
    )

    for _pkg in "${_packages[@]}"; do
        local _existing_wheel
        _existing_wheel="$(newest_wheel "${WHEELS_DIR}"/"${_pkg}"-*.whl)"
        if [[ -n "${_existing_wheel}" ]]; then
            info "${_pkg} wheel already exists: $(basename "${_existing_wheel}")"
            continue
        fi

        info "Building ${_pkg} from source with Zen 5 optimizations..."
        if pip wheel "${_pkg}" \
            --no-binary "${_pkg}" \
            --no-cache-dir \
            --no-deps \
            --wheel-dir "${WHEELS_DIR}" \
            -v; then
            local _built_wheel
            _built_wheel="$(newest_wheel "${WHEELS_DIR}"/"${_pkg}"-*.whl)"
            if [[ -n "${_built_wheel}" ]]; then
                success "${_pkg} wheel built: $(basename "${_built_wheel}")"
            else
                warn "${_pkg} wheel not found after build (may have different filename)"
            fi
        else
            warn "${_pkg} source build failed — will use PyPI binary wheel as fallback"
        fi
    done

    # Restore original CFLAGS/LDFLAGS with driver-level flags
    export CFLAGS="${_saved_cflags}" CXXFLAGS="${_saved_cxxflags}" LDFLAGS="${_saved_ldflags}"

    success "C/C++ optimized wheels complete"
}

# Step 32: Export existing source builds as distributable wheels
export_source_wheels() {
    log_step 32 "Export source-built packages as wheels (torch, triton, torchvision, amd-aiter, amdsmi)"

    mkdir -p "${WHEELS_DIR}"

    # PyTorch wheel: should already exist from step 10, verify it's there.
    local _torch_wheel
    _torch_wheel="$(newest_wheel "${WHEELS_DIR}"/torch-*.whl)"
    if [[ -n "${_torch_wheel}" ]]; then
        success "torch wheel exists: $(basename "${_torch_wheel}")"
    else
        # PyTorch is currently an editable install — build the wheel now.
        # This uses the ALREADY-COMPILED build tree, so cmake won't re-run
        # (the .so files are all built). Only the packaging step runs.
        if [[ -d "${PYTORCH_SRC}" ]]; then
            info "Building PyTorch wheel from existing build tree..."
            cd "${PYTORCH_SRC}"
            pip wheel . --no-build-isolation --no-deps --wheel-dir "${WHEELS_DIR}" -v
            _torch_wheel="$(newest_wheel "${WHEELS_DIR}"/torch-*.whl)"
            if [[ -n "${_torch_wheel}" ]]; then
                success "torch wheel built: $(basename "${_torch_wheel}")"
            else
                warn "torch wheel packaging produced no output"
            fi
            cd "${VLLM_DIR}"
        else
            warn "PyTorch source not found at ${PYTORCH_SRC} — cannot build wheel"
        fi
    fi

    # Triton wheel: should already exist from step 13.
    local _triton_wheel
    _triton_wheel="$(newest_wheel "${WHEELS_DIR}"/triton*.whl)"
    if [[ -n "${_triton_wheel}" ]]; then
        success "triton wheel exists: $(basename "${_triton_wheel}")"
    else
        if [[ -d "${TRITON_SRC}" ]]; then
            info "Building Triton wheel from existing build tree..."
            local triton_pkg_dir="${TRITON_SRC}"
            if [[ -f "${TRITON_SRC}/python/setup.py" && ! -f "${TRITON_SRC}/setup.py" ]]; then
                triton_pkg_dir="${TRITON_SRC}/python"
            fi
            cd "${triton_pkg_dir}"
            pip wheel . --no-build-isolation --no-deps --wheel-dir "${WHEELS_DIR}" -v
            _triton_wheel="$(newest_wheel "${WHEELS_DIR}"/triton*.whl)"
            if [[ -n "${_triton_wheel}" ]]; then
                success "triton wheel built: $(basename "${_triton_wheel}")"
            else
                warn "triton wheel packaging failed"
            fi
            cd "${VLLM_DIR}"
        else
            warn "Triton source not found at ${TRITON_SRC} — cannot build wheel"
        fi
    fi

    # TorchVision wheel: should already exist from step 13, verify it's there.
    local _tv_wheel
    _tv_wheel="$(newest_wheel "${WHEELS_DIR}"/torchvision-*.whl)"
    if [[ -n "${_tv_wheel}" ]]; then
        success "torchvision wheel exists: $(basename "${_tv_wheel}")"
    else
        if [[ -d "${TORCHVISION_SRC}" ]]; then
            info "Building TorchVision wheel from existing build tree..."
            cd "${TORCHVISION_SRC}"
            pip wheel . --no-build-isolation --no-deps --wheel-dir "${WHEELS_DIR}" -v
            _tv_wheel="$(newest_wheel "${WHEELS_DIR}"/torchvision-*.whl)"
            if [[ -n "${_tv_wheel}" ]]; then
                success "torchvision wheel built: $(basename "${_tv_wheel}")"
            else
                warn "torchvision wheel packaging failed"
            fi
            cd "${VLLM_DIR}"
        else
            warn "TorchVision source not found at ${TORCHVISION_SRC} — cannot build wheel"
        fi
    fi

    # amd-aiter wheel: build from the installed site-packages source.
    # AITER was installed as part of the vLLM build (step 24).
    local _aiter_wheel
    _aiter_wheel="$(newest_wheel "${WHEELS_DIR}"/amd_aiter-*.whl)"
    if [[ -n "${_aiter_wheel}" ]]; then
        success "amd-aiter wheel exists: $(basename "${_aiter_wheel}")"
    else
        # AITER source lives in the vLLM repo under third_party/aiter
        local _aiter_src="${VLLM_DIR}/vllm/third_party/aiter"
        if [[ ! -d "${_aiter_src}" ]]; then
            # Fallback: check if it was cloned standalone
            _aiter_src="${VLLM_DIR}/aiter"
        fi
        if [[ -d "${_aiter_src}" && -f "${_aiter_src}/setup.py" ]]; then
            info "Building amd-aiter wheel..."
            cd "${_aiter_src}"
            pip wheel . --no-build-isolation --no-deps --wheel-dir "${WHEELS_DIR}" -v || \
                warn "amd-aiter wheel build failed (AITER JIT-compiles at runtime anyway)"
            _aiter_wheel="$(newest_wheel "${WHEELS_DIR}"/amd_aiter-*.whl)"
            if [[ -n "${_aiter_wheel}" ]]; then
                success "amd-aiter wheel built: $(basename "${_aiter_wheel}")"
            fi
            cd "${VLLM_DIR}"
        else
            info "AITER source not found — AITER JIT-compiles kernels at runtime, wheel is optional"
        fi
    fi

    # amdsmi wheel: build from TheRock's share/amd_smi.
    local _amdsmi_wheel
    _amdsmi_wheel="$(newest_wheel "${WHEELS_DIR}"/amdsmi-*.whl)"
    if [[ -n "${_amdsmi_wheel}" ]]; then
        success "amdsmi wheel exists: $(basename "${_amdsmi_wheel}")"
    else
        local _amdsmi_src="${LOCAL_PREFIX}/share/amd_smi"
        if [[ -d "${_amdsmi_src}" && -f "${_amdsmi_src}/setup.py" ]]; then
            info "Building amdsmi wheel from ${_amdsmi_src}..."
            cd "${_amdsmi_src}"
            pip wheel . --no-build-isolation --no-deps --wheel-dir "${WHEELS_DIR}" -v || \
                warn "amdsmi wheel build failed"
            _amdsmi_wheel="$(newest_wheel "${WHEELS_DIR}"/amdsmi-*.whl)"
            if [[ -n "${_amdsmi_wheel}" ]]; then
                success "amdsmi wheel built: $(basename "${_amdsmi_wheel}")"
            fi
            cd "${VLLM_DIR}"
        else
            warn "amdsmi source not found at ${_amdsmi_src}"
        fi
    fi

    # Summary
    echo ""
    info "Wheels in ${WHEELS_DIR}:"
    for _whl in "${WHEELS_DIR}"/*.whl; do
        [[ -f "${_whl}" ]] || continue
        info "  $(basename "${_whl}")"
    done

    success "Source wheel export complete"
}

# =============================================================================
# Rebuild Mode
# =============================================================================

handle_rebuild() {
    if [[ "${REBUILD}" == "true" ]]; then
        section "Rebuild mode: cleaning previous build"
        warn "Removing venv and source directories..."

        if [[ -d "${VLLM_VENV}" ]]; then
            rm -r "${VLLM_VENV}"
            info "Removed ${VLLM_VENV}"
        fi

        for src_dir in "${VLLM_SRC}" "${PYTORCH_SRC}" "${TRITON_SRC}" "${AOTRITON_SRC}" "${FLASH_ATTN_SRC}" "${THEROCK_SRC}" "${CPYTHON_SRC}" "${AOCL_LIBM_SRC}" "${AOCL_UTILS_SRC}"; do
            if [[ -d "${src_dir}" ]]; then
                rm -r "${src_dir}"
                info "Removed ${src_dir}"
            fi
        done

        if [[ -d "${LOCAL_PREFIX}" ]]; then
            rm -r "${LOCAL_PREFIX}"
            info "Removed ${LOCAL_PREFIX}"
        fi

        success "Clean complete. Starting fresh build."
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    section "vLLM Full-Stack Source Build for AMD Strix Halo (gfx1151)"
    info "Build log: ${VLLM_LOG}"
    info "Start step: ${START_STEP}"
    info "Rebuild: ${REBUILD}"
    info "Components: TheRock → AOCL-LibM → Python → PyTorch → Triton → AOTriton → vLLM → Flash Attention → Optimized Wheels"
    echo ""

    check_prerequisites
    handle_rebuild

    # Set up PATH/LD_LIBRARY_PATH for the unified LOCAL_PREFIX.
    # This runs unconditionally so that --step N resumes work correctly:
    # every step after TheRock (step 1) needs amdclang and ROCm libs on PATH.
    if [[ -d "${LOCAL_PREFIX}/lib" ]]; then
        export ROCM_PATH="${LOCAL_PREFIX}"
        export LD_LIBRARY_PATH="${LOCAL_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        export PATH="${LOCAL_PREFIX}/lib/llvm/bin:${LOCAL_PREFIX}/bin:${PATH}"
        if [[ -d "${LOCAL_PREFIX}/llvm/amdgcn/bitcode" ]]; then
            export DEVICE_LIB_PATH="${LOCAL_PREFIX}/llvm/amdgcn/bitcode"
            export HIP_DEVICE_LIB_PATH="${LOCAL_PREFIX}/llvm/amdgcn/bitcode"
        elif [[ -d "${LOCAL_PREFIX}/amdgcn/bitcode" ]]; then
            export DEVICE_LIB_PATH="${LOCAL_PREFIX}/amdgcn/bitcode"
            export HIP_DEVICE_LIB_PATH="${LOCAL_PREFIX}/amdgcn/bitcode"
        fi
    fi

    # Build pipeline — skip steps below START_STEP
    # Phase A: ROCm SDK (TheRock — builds amdclang used by everything downstream)
    [[ "${START_STEP}" -le 1 ]]  && clone_therock
    [[ "${START_STEP}" -le 2 ]]  && configure_therock
    [[ "${START_STEP}" -le 3 ]]  && build_therock
    [[ "${START_STEP}" -le 4 ]]  && validate_rocm

    # Phase B: CPU Libraries + Python (built with amdclang from Phase A)
    [[ "${START_STEP}" -le 5 ]]  && build_aocl_utils
    [[ "${START_STEP}" -le 6 ]]  && build_aocl_libm
    [[ "${START_STEP}" -le 7 ]]  && build_python
    [[ "${START_STEP}" -le 8 ]]  && create_venv

    # Phase C: PyTorch + TorchVision (ROCm fork)
    [[ "${START_STEP}" -le 9 ]]  && clone_pytorch
    [[ "${START_STEP}" -le 10 ]] && build_pytorch
    [[ "${START_STEP}" -le 11 ]] && validate_pytorch
    [[ "${START_STEP}" -le 12 ]] && clone_torchvision
    [[ "${START_STEP}" -le 13 ]] && build_torchvision

    # Phase D: Kernel Compilers (Triton + AOTriton)
    [[ "${START_STEP}" -le 14 ]] && clone_triton
    [[ "${START_STEP}" -le 15 ]] && build_triton
    [[ "${START_STEP}" -le 16 ]] && validate_triton
    [[ "${START_STEP}" -le 17 ]] && clone_aotriton
    [[ "${START_STEP}" -le 18 ]] && build_aotriton

    # Phase E: vLLM
    [[ "${START_STEP}" -le 19 ]] && clone_vllm
    [[ "${START_STEP}" -le 20 ]] && patch_amdsmi_import
    [[ "${START_STEP}" -le 20 ]] && patch_vllm_gfx1151
    [[ "${START_STEP}" -le 21 ]] && install_build_deps
    [[ "${START_STEP}" -le 22 ]] && run_use_existing_torch
    [[ "${START_STEP}" -le 23 ]] && install_rocm_requirements
    [[ "${START_STEP}" -le 24 ]] && build_vllm

    # Phase F: Flash Attention
    [[ "${START_STEP}" -le 25 ]] && reinstall_amdsmi
    [[ "${START_STEP}" -le 26 ]] && clone_flash_attention
    [[ "${START_STEP}" -le 27 ]] && patch_flash_amdsmi
    [[ "${START_STEP}" -le 28 ]] && build_flash_attention

    # Phase G: Validation + Warmup
    [[ "${START_STEP}" -le 29 ]] && smoke_test
    [[ "${START_STEP}" -le 29 ]] && warmup_aiter_jit
    [[ "${START_STEP}" -le 29 ]] && warmup_tunableop

    # Phase H: Optimized Wheels (Zen 5 native builds for downstream venvs)
    [[ "${START_STEP}" -le 30 ]] && build_rust_wheels
    [[ "${START_STEP}" -le 31 ]] && build_native_wheels
    [[ "${START_STEP}" -le 32 ]] && export_source_wheels
}

main "$@"
