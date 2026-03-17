#!/usr/bin/env bash
# Copyright 2026 Blackcat Informatics Inc.
# SPDX-License-Identifier: MIT
#
# vllm-env.sh - Environment activation for vLLM source builds
#
# This script is designed to be SOURCED, not executed:
#   source scripts/vllm-env.sh
#
# It sets compiler flags, ROCm paths, and activates the vLLM venv.
# Safe to source multiple times (idempotent).
#
# Prerequisites:
#   - Clang 21+ installed
#   - /opt/src/vllm/ directory exists (created by build-vllm.sh)
#
# Usage:
#   source scripts/vllm-env.sh              # Activate environment
#   source scripts/vllm-env.sh --info       # Show current settings

# Guard: this file must be sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced, not executed."
    echo "Usage: source ${BASH_SOURCE[0]}"
    exit 1
fi

# =============================================================================
# Base Directories
# =============================================================================

export VLLM_DIR="/opt/src/vllm"
export VLLM_VENV="${VLLM_DIR}/.venv"
export VLLM_SRC="${VLLM_DIR}/vllm"
export VLLM_LOG="${VLLM_DIR}/build.log"

# =============================================================================
# Unified Install Prefix
# =============================================================================
# All C/C++ components (AOCL-Utils, AOCL-LibM, TheRock ROCm, AOTriton, CPython)
# install to a single prefix mirroring /usr/local/ layout.

_LOCAL_PREFIX="${VLLM_DIR}/local"

# AOCL-LibM provides Zen 5 optimized transcendental functions (exp, log, sin, etc.)
# that replace glibc libm. Built with AVX-512 paths for native 512-bit execution.
_AOCL_LDFLAGS=""
if [[ -f "${_LOCAL_PREFIX}/lib/libalm.so" ]]; then
    _AOCL_LDFLAGS="-L${_LOCAL_PREFIX}/lib -lalm"
fi

# =============================================================================
# Compiler Toolchain
# =============================================================================
# After TheRock builds, use its AMD-forked Clang which includes -famd-opt
# (AMD proprietary Zen optimizations). Falls back to system clang if TheRock
# hasn't been built yet.

_THEROCK_CLANG="${_LOCAL_PREFIX}/lib/llvm/bin/amdclang"
if [[ -x "${_THEROCK_CLANG}" ]]; then
    export CC="${_THEROCK_CLANG}"
    export CXX="${_LOCAL_PREFIX}/lib/llvm/bin/amdclang++"
    _AMD_OPT="-famd-opt"
else
    export CC=clang
    export CXX=clang++
    _AMD_OPT=""
fi

# Polly (polyhedral loop optimizer): enabled if the built clang supports it.
# Polly restructures loop nests for cache locality — critical for Strix Halo's
# LPDDR5X unified memory hierarchy where cache misses are expensive.
_POLLY_FLAGS=""
if "${CC}" -mllvm -polly -x c /dev/null -c -o /dev/null 2>/dev/null; then
    _POLLY_FLAGS="-mllvm -polly -mllvm -polly-vectorizer=stripmine"
fi

# Full "power user" flag set for AMD Strix Halo (Zen 5 + RDNA 3.5).
# All flags derive from _BASE_CFLAGS to eliminate duplication (DRY).
#
# Flag rationale:
#   -O3 -march=native:       Target Zen 5 microarchitecture natively
#   -flto=thin:              Parallel LTO for 16-core design
#   -mprefer-vector-width=512: Native 512-bit AVX-512 (no downclock on Zen 5)
#   -mavx512*:               Explicit AVX-512 subsets for Zen 5
#   -famd-opt:               AMD proprietary Zen microarch tuning (amdclang only)
#   -mllvm -polly:           Polyhedral loop optimizer (cache-locality restructuring)
#   -mllvm -polly-vectorizer=stripmine: Strip-mine loops for Strix Halo cache hierarchy
#   -mllvm -inline-threshold=600: Aggressive inlining for Zen 5's wide issue pipeline
#   -mllvm -unroll-threshold=150: Aggressive unrolling for Zen 5's large reorder buffer
#   -mllvm -adce-remove-loops: Clean up dead loop structures in AI/scientific code
#   -Wno-error=unused-command-line-argument: Prevent -famd-opt and -mllvm flags from
#                             becoming fatal errors when passed through to link steps
#                             or translation units where they don't apply (e.g. googletest)
_BASE_CFLAGS="-O3 -DNDEBUG -march=native -flto=thin -mprefer-vector-width=512 -mavx512f -mavx512dq -mavx512vl -mavx512bw ${_AMD_OPT} ${_POLLY_FLAGS} -mllvm -inline-threshold=600 -mllvm -unroll-threshold=150 -mllvm -adce-remove-loops -Wno-error=unused-command-line-argument"
_BASE_LDFLAGS="-flto=thin -fuse-ld=lld -Wl,-rpath,${_LOCAL_PREFIX}/lib ${_AOCL_LDFLAGS}"

# Autotools / setup.py: read CFLAGS, CXXFLAGS, LDFLAGS from environment.
export CFLAGS="${_BASE_CFLAGS}"
export CXXFLAGS="${_BASE_CFLAGS}"
export LDFLAGS="${_BASE_LDFLAGS}"

# CMake builds: cmake ignores CFLAGS/CXXFLAGS by default. PyTorch's setup.py
# (tools/setup_helpers/cmake.py line 318) auto-forwards any env var starting
# with CMAKE_ as -D defines. Other cmake projects pick these up directly.
# Without these, cmake uses bare "-O3 -DNDEBUG" and none of our Zen 5 flags
# reach the compiler.
export CMAKE_C_FLAGS_RELEASE="${_BASE_CFLAGS}"
export CMAKE_CXX_FLAGS_RELEASE="${_BASE_CFLAGS}"
export CMAKE_EXE_LINKER_FLAGS="${_BASE_LDFLAGS}"
export CMAKE_SHARED_LINKER_FLAGS="${_BASE_LDFLAGS}"

unset _BASE_CFLAGS _BASE_LDFLAGS _POLLY_FLAGS _AOCL_LDFLAGS _THEROCK_CLANG _AMD_OPT

# =============================================================================
# ROCm / GPU Configuration (RDNA 3.5, gfx1151)
# =============================================================================

export PYTORCH_ROCM_ARCH="gfx1151"
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export HSA_OVERRIDE_GFX_VERSION="11.5.1"

# HIP GPU compiler flags for gfx1151 (Strix Halo RDNA 3.5 iGPU, 40 CUs)
# --offload-arch=gfx1151:              Target the integrated RDNA 3.5 GPU
# -amdgpu-early-inline-all=true:       Aggressively inline GPU kernel functions for
#                                       better register allocation on RDNA 3.5
# -amdgpu-function-calls=false:        Eliminate function calls entirely via inlining;
#                                       on the iGPU, call/return stalls the wavefront
# -famd-opt:                           AMD proprietary GPU microarch tuning
export HIP_CLANG_FLAGS="--offload-arch=gfx1151 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -famd-opt"

# ROCm path: source build installs to the unified local/ prefix.
# build-vllm.sh writes the git version tag to .rocm-version for reference.
_ROCM_VERSION_FILE="${VLLM_DIR}/.rocm-version"

if [[ -d "${_LOCAL_PREFIX}/lib" ]]; then
    # Source-compiled TheRock (preferred — unified prefix)
    export ROCM_PATH="${_LOCAL_PREFIX}"
elif [[ -f "${_ROCM_VERSION_FILE}" ]]; then
    # Legacy: tarball-extracted ROCm
    ROCM_VERSION="$(cat "${_ROCM_VERSION_FILE}")"
    export ROCM_VERSION
    export ROCM_PATH="${VLLM_DIR}/rocm-${ROCM_VERSION}"
fi

if [[ -n "${ROCM_PATH:-}" && -d "${ROCM_PATH}" ]]; then
    export LD_LIBRARY_PATH="${ROCM_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export PATH="${ROCM_PATH}/lib/llvm/bin:${ROCM_PATH}/bin:${PATH}"

    # Device bitcode may be under llvm/ or directly under amdgcn/
    if [[ -d "${ROCM_PATH}/llvm/amdgcn/bitcode" ]]; then
        export DEVICE_LIB_PATH="${ROCM_PATH}/llvm/amdgcn/bitcode"
        export HIP_DEVICE_LIB_PATH="${ROCM_PATH}/llvm/amdgcn/bitcode"
    elif [[ -d "${ROCM_PATH}/amdgcn/bitcode" ]]; then
        export DEVICE_LIB_PATH="${ROCM_PATH}/amdgcn/bitcode"
        export HIP_DEVICE_LIB_PATH="${ROCM_PATH}/amdgcn/bitcode"
    fi
fi

unset _ROCM_VERSION_FILE

# =============================================================================
# AITER Configuration
# =============================================================================
# AITER (AMD Inference Triton Engine for ROCm) provides optimized attention,
# linear, MoE, and RMSNorm kernels. The AMD fork has full gfx1151 support
# (chip_info.py enum 13, fwd_prefill.py RDNA 3.5 tuning: BLOCK_M=32,
# BLOCK_N=32, waves_per_eu=2). Build status recorded by build-vllm.sh.

# CK_DIR: Composable Kernel source tree for AITER JIT compilation.
# AITER's MHA kernels (mha_varlen_fwd, mha_fwd_splitkv) require CK tile headers
# and the generate.py codegen script at runtime. The pip-installed aiter wheel
# does NOT include the CK 3rdparty submodule — only the ck_helper shim.
# We point CK_DIR to the CK source shipped with PyTorch's AITER 3rdparty,
# which has the complete example/ck_tile/01_fmha/ codegen (generate.py,
# fmha_fwd.hpp, mask.hpp).
_CK_PYTORCH="${VLLM_DIR}/pytorch/third_party/aiter/3rdparty/composable_kernel"
_CK_THEROCK="${VLLM_DIR}/therock/rocm-libraries/projects/composablekernel"
if [[ -d "${_CK_PYTORCH}/example/ck_tile/01_fmha" ]]; then
    export CK_DIR="${_CK_PYTORCH}"
elif [[ -d "${_CK_THEROCK}/example/ck_tile/01_fmha" ]]; then
    export CK_DIR="${_CK_THEROCK}"
fi
unset _CK_PYTORCH _CK_THEROCK

_AITER_STATUS_FILE="${VLLM_DIR}/.aiter-status"
if [[ -f "${_AITER_STATUS_FILE}" ]]; then
    _AITER_STATUS="$(cat "${_AITER_STATUS_FILE}")"
    if [[ "${_AITER_STATUS}" == "enabled" ]]; then
        # Master switch: enables AITER backend selection in vLLM
        export VLLM_ROCM_USE_AITER=1

        # Sub-feature flags: set explicitly so they appear in --info output and
        # are auditable. These control which AITER-optimized operators are used.
        # All verified working on gfx1151 via direct kernel execution tests.
        export VLLM_ROCM_USE_AITER_LINEAR=1
        export VLLM_ROCM_USE_AITER_MOE=1
        export VLLM_ROCM_USE_AITER_RMSNORM=1
        # AITER MHA (CK tile backend): dispatches through flash_attn_varlen_func
        # → mha_varlen_fwd, a CK-tile kernel JIT-compiled at runtime.
        # Requires aiter wheel built against matching CK source (see CK_DIR).
        export VLLM_ROCM_USE_AITER_MHA=1
        export VLLM_ROCM_USE_AITER_TRITON_GEMM=1

        # Triton fused RoPE + KV cache update — verified on gfx1151
        # (aiter.ops.triton.fused_kv_cache.fused_qk_rope_reshape_and_cache)
        export VLLM_ROCM_USE_AITER_TRITON_ROPE=1

        # Unified attention: dispatches through AITER's CK-tile MHA path.
        # vLLM envs.py defaults this to False; we enable it explicitly.
        export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1

        # Fused shared expert MoE — uses AITER native fmoe_g1u1 kernels.
        # Only activates for DeepSeek-style models with shared experts.
        export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1

        # Shuffle KV cache layout — DISABLED: pa_fwd_asm assembly kernel
        # dispatch table lacks some dtype/GQA/block_size combos on gfx1151,
        # causing "cannot get heuristic kernel" at runtime. Needs AITER
        # tuning table update for RDNA 3.5 before enabling.
        # export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
    else
        unset VLLM_ROCM_USE_AITER 2>/dev/null || true
        unset VLLM_ROCM_USE_AITER_LINEAR 2>/dev/null || true
        unset VLLM_ROCM_USE_AITER_MOE 2>/dev/null || true
        unset VLLM_ROCM_USE_AITER_RMSNORM 2>/dev/null || true
        unset VLLM_ROCM_USE_AITER_MHA 2>/dev/null || true
        unset VLLM_ROCM_USE_AITER_TRITON_GEMM 2>/dev/null || true
        unset VLLM_ROCM_USE_AITER_TRITON_ROPE 2>/dev/null || true
        unset VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION 2>/dev/null || true
        unset VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS 2>/dev/null || true
    fi
    unset _AITER_STATUS
fi
unset _AITER_STATUS_FILE

# =============================================================================
# PyTorch Runtime Optimization
# =============================================================================
# These flags optimize PyTorch's BLAS backend and GEMM dispatch for Strix Halo.
# They require no rebuild — they control runtime behavior only.

# hipBLASLt: AMD's high-performance BLAS library. On gfx1151 (ROCM_VERSION >= 70000),
# hipBLASLt is available and includes gfx1151 in its supported architectures
# (HIPHooks.cpp). Without this flag, PyTorch defaults to rocBLAS which lacks
# some fused GEMM kernels that hipBLASLt provides.
export TORCH_BLAS_PREFER_HIPBLASLT=1

# TunableOp: PyTorch's runtime GEMM autotuning. On first run, benchmarks multiple
# GEMM implementations (rocBLAS, hipBLASLt, CK) for each unique problem shape
# and records the fastest. Subsequent runs use the cached results. The CSV file
# persists tuning data across vLLM restarts — critical for Strix Halo where the
# default kernel selection often picks suboptimal shapes for the 40-CU iGPU.
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_FILENAME="${VLLM_DIR}/tunableop_results_gfx11510.csv"

# torch.inductor codegen bug: AttrsDescriptor.__repr__() in triton/backends/
# compiler.py produces invalid Python (angle-bracket object repr) in generated
# Triton source files. Fixed by build-vllm.sh Patch 5 which adds a proper
# __repr__ using the existing to_dict()/from_dict() round-trip.
# TORCH_COMPILE_DISABLE was previously set here as a workaround for Triton
# instruction scheduler bugs on gfx1151. With Patch 5 (AttrsDescriptor) and
# Patch 6 (duplicate pattern registration) in build-vllm.sh, torch.compile
# with Inductor now works correctly on gfx1151 — all AITER optimizations
# active, verified with Qwen2.5-7B-Instruct inference.
unset TORCH_COMPILE_DISABLE 2>/dev/null || true

# AOTriton experimental: enables gfx11xx experimental kernel paths in the
# AOTriton (Ahead-Of-Time Triton) library. Required for Strix Halo gfx1151.
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# =============================================================================
# MIOpen Configuration
# =============================================================================
# MIOpen (AMD's convolution/GEMM library) algorithm selection strategy.
# MIOPEN_FIND_MODE controls how MIOpen searches for optimal kernels:
#   1 = NORMAL:      Exhaustive search (very slow on first run per shape)
#   2 = FAST:        Heuristic-only (fast startup, good enough with TunableOp)
#   3 = HYBRID:      Check find-db, fall back to exhaustive search on miss
#   4 = FAST_HYBRID: Check find-db, fall back to heuristic on miss
#
# We use FAST because TunableOp already handles GEMM autotuning at the PyTorch
# level. MIOpen's exhaustive search is redundant and adds minutes to CUDA graph
# warmup (51 batch sizes × multiple kernel shapes per size).
export MIOPEN_FIND_MODE=2

# =============================================================================
# Triton Compiler Optimization
# =============================================================================
# Triton JIT compiler flags that affect kernel code generation for gfx11.

# Buffer ops: instructs the Triton compiler to use buffer (global memory) load/store
# operations instead of flat operations on gfx11. Buffer ops have better memory
# coalescing behavior on RDNA 3.5's memory controller, particularly for the
# strided access patterns common in attention and GEMM kernels.
export AMDGCN_USE_BUFFER_OPS=1

# =============================================================================
# Rust Compiler Flags (Zen 5)
# =============================================================================
# target-cpu=znver5 explicitly (not 'native') because Rust's native detection
# has a bug where it identifies znver5 but only enables SSE2. Explicit znver5
# gives us all 40+ target features: AVX-512{F,BW,DQ,VL,VNNI,IFMA,VBMI,VBMI2,
# BITALG,BF16,VPOPCNTDQ,VP2INTERSECT}, VAES, VPCLMULQDQ, GFNI, SHA.
#
# Do NOT add -C lto=thin — maturin adds -C embed-bitcode=no which conflicts.
export RUSTFLAGS="-C target-cpu=znver5 -C opt-level=3"

# =============================================================================
# AOTriton Install Prefix
# =============================================================================
# AOTriton installs to the unified LOCAL_PREFIX alongside TheRock and AOCL.
# PyTorch's cmake discovery and vLLM's build both need this to find
# libaotriton.so and the AOTriton cmake config.
export AOTRITON_INSTALL_DIR="${_LOCAL_PREFIX}"

# =============================================================================
# Lemonade / llama.cpp Configuration
# =============================================================================
# Lemonade wraps llama.cpp (GPU/CPU), FLM (NPU), and ONNX behind an
# OpenAI-compatible API. The llama-server binary lives at
# ${VLLM_VENV}/rocm/llama_server/ with a .env file for runtime optimizations.

# ROCm backend (primary — hipBLAS, best prefill <32K context)
_LLAMACPP_ROCM="${VLLM_VENV}/rocm/llama_server"
if [[ -d "${_LLAMACPP_ROCM}" ]]; then
    export LEMONADE_LLAMACPP_DIR="${_LLAMACPP_ROCM}"
    export PATH="${_LLAMACPP_ROCM}:${PATH}"
    export LD_LIBRARY_PATH="${_LLAMACPP_ROCM}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Vulkan backend (best generation speed + prefill >32K context on gfx1151)
_LLAMACPP_VULKAN="${VLLM_VENV}/vulkan/llama_server"
if [[ -d "${_LLAMACPP_VULKAN}" ]]; then
    export LEMONADE_LLAMACPP_VULKAN_DIR="${_LLAMACPP_VULKAN}"
    export LD_LIBRARY_PATH="${_LLAMACPP_VULKAN}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
unset _LLAMACPP_ROCM _LLAMACPP_VULKAN

# =============================================================================
# Virtual Environment Activation
# =============================================================================

if [[ -d "${VLLM_VENV}" ]]; then
    # shellcheck source=/dev/null
    source "${VLLM_VENV}/bin/activate"
fi

# =============================================================================
# Info Display (--info flag)
# =============================================================================

if [[ "${1:-}" == "--info" ]]; then
    echo "vLLM Build Environment"
    echo "======================"
    echo ""
    echo "  Directories:"
    echo "    VLLM_DIR:         ${VLLM_DIR}"
    echo "    VLLM_VENV:        ${VLLM_VENV}"
    echo "    LOCAL_PREFIX:     ${_LOCAL_PREFIX}"
    echo ""
    echo "  Compiler:"
    echo "    CC:               ${CC}"
    echo "    CXX:              ${CXX}"
    echo "    CFLAGS:           ${CFLAGS}"
    echo "    LDFLAGS:          ${LDFLAGS}"
    echo ""
    echo "  GPU / ROCm:"
    echo "    ROCM_ARCH:        ${PYTORCH_ROCM_ARCH}"
    echo "    ROCM_PATH:        ${ROCM_PATH:-<not set -- run build-vllm.sh first>}"
    echo "    ROCM_VERSION:     ${ROCM_VERSION:-<not set>}"
    echo "    HIP_CLANG_FLAGS:  ${HIP_CLANG_FLAGS}"
    echo "    HSA_GFX_OVERRIDE: ${HSA_OVERRIDE_GFX_VERSION}"
    echo ""
    echo "  Components:"
    echo "    AOCL-LibM:        $([[ -f "${_LOCAL_PREFIX}/lib/libalm.so" ]] && echo 'installed' || echo 'not built')"
    echo "    AOTriton:         $([[ -d "${_LOCAL_PREFIX}/lib/cmake/aotriton" ]] && echo 'installed' || echo 'not built')"
    echo "    Flash Attn Triton: ${FLASH_ATTENTION_TRITON_AMD_ENABLE}"
    echo ""
    echo "  AITER Optimization:"
    echo "    CK_DIR:          ${CK_DIR:-<not set — MHA JIT will fail>}"
    echo "    Master switch:    ${VLLM_ROCM_USE_AITER:-disabled}"
    echo "    Linear layers:    ${VLLM_ROCM_USE_AITER_LINEAR:-default}"
    echo "    MoE kernels:      ${VLLM_ROCM_USE_AITER_MOE:-default}"
    echo "    RMSNorm:          ${VLLM_ROCM_USE_AITER_RMSNORM:-default}"
    echo "    MHA (CK tile):    ${VLLM_ROCM_USE_AITER_MHA:-disabled (CK ABI mismatch)}"
    echo "    Triton GEMM:      ${VLLM_ROCM_USE_AITER_TRITON_GEMM:-default}"
    echo "    Triton ROPE:      ${VLLM_ROCM_USE_AITER_TRITON_ROPE:-disabled}"
    echo "    Unified attn:     ${VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION:-disabled}"
    echo "    Shared expert MoE:${VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS:-disabled}"
    echo "    Shuffle KV cache: ${VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT:-disabled (pa_fwd_asm tuning gap)}"
    echo ""
    echo "  PyTorch Runtime:"
    echo "    hipBLASLt:        ${TORCH_BLAS_PREFER_HIPBLASLT:-disabled}"
    echo "    TunableOp:        ${PYTORCH_TUNABLEOP_ENABLED:-disabled}"
    echo "    TunableOp file:   ${PYTORCH_TUNABLEOP_FILENAME:-<not set>}"
    echo ""
    echo "  Rust:"
    echo "    RUSTFLAGS:        ${RUSTFLAGS}"
    echo ""
    echo "  Triton Compiler:"
    echo "    Buffer ops:       ${AMDGCN_USE_BUFFER_OPS:-disabled}"
    echo ""
    echo "  Lemonade / llama.cpp:"
    echo "    ROCm backend:     ${LEMONADE_LLAMACPP_DIR:-<not built>}"
    echo "    Vulkan backend:   ${LEMONADE_LLAMACPP_VULKAN_DIR:-<not built>}"
    echo "    llama-server:     $(command -v llama-server 2>/dev/null || echo 'not in PATH')"
    echo "    llama-bench:      $(command -v llama-bench 2>/dev/null || echo 'not in PATH')"
    echo "    llama-quantize:   $(command -v llama-quantize 2>/dev/null || echo 'not in PATH')"
    if [[ -f "${LEMONADE_LLAMACPP_DIR:-.}/.env" ]]; then
        echo "    ROCm .env:        present"
    else
        echo "    ROCm .env:        not found"
    fi
    if [[ -f "${LEMONADE_LLAMACPP_VULKAN_DIR:-.}/.env" ]]; then
        echo "    Vulkan .env:      present"
    else
        echo "    Vulkan .env:      not found"
    fi
    echo ""
    echo "  Runtime:"
    echo "    VENV active:      $(command -v python 2>/dev/null || echo 'no')"
fi

unset _LOCAL_PREFIX
