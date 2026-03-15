#!/usr/bin/env bash
# Copyright 2026 Blackcat Informatics Inc.
# SPDX-License-Identifier: MIT
#
# vllm-start.sh - Start all vLLM inference instances as background processes
#
# Launches one vLLM process per role defined in VLLM_ROLES. Each role gets
# its own port, device, PID file, and log file. All instances are managed
# as a single logical server.
#
# Per-role configuration is read from .env using the convention:
#   VLLM_<ROLE>_MODEL              - HuggingFace model ID (required)
#   VLLM_<ROLE>_PORT               - Listen port (required)
#   VLLM_<ROLE>_DEVICE             - Device type: rocm, cpu (required)
#   VLLM_<ROLE>_GPU_MEMORY_MB      - GPU memory limit in MB (optional)
#   VLLM_<ROLE>_ATTENTION_BACKEND  - Override attention backend (optional)
#   VLLM_<ROLE>_EXTRA_ARGS         - Additional vLLM CLI args (optional)
#
# Prerequisites:
#   - .env in the repo root with VLLM_* configuration
#   - vLLM built and available in PATH (via vllm-env.sh)
#   - ROCm installed for GPU roles
#
# Usage:
#   scripts/vllm-start.sh

set -euo pipefail

# =============================================================================
# Setup
# =============================================================================

_SCRIPT_REAL_PATH="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || realpath "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")"
_SCRIPT_DIR="$(cd "$(dirname "$_SCRIPT_REAL_PATH")" && pwd)"

# Portable: try platform layout, fall back to co-located files.
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

# shellcheck source=vllm-env.sh disable=SC1091
source "${_SCRIPT_DIR}/vllm-env.sh"

_HELPERS="${_SCRIPT_DIR}/../lib/sh/vllm-runtime-helpers.sh"
[[ -f "${_HELPERS}" ]] || _HELPERS="${_SCRIPT_DIR}/vllm-runtime-helpers.sh"
# shellcheck source=../lib/sh/vllm-runtime-helpers.sh
source "${_HELPERS}"
unset _HELPERS

PLATFORM_DIR="${_SCRIPT_DIR}/.."
[[ -d "${PLATFORM_DIR}/lib" ]] || PLATFORM_DIR="${_SCRIPT_DIR}"
ENV_FILE="${PLATFORM_DIR}/../.env"
[[ -f "${ENV_FILE}" ]] || ENV_FILE="${_SCRIPT_DIR}/.env"

unset _SCRIPT_REAL_PATH _SCRIPT_DIR

# Load .env configuration.
vllm_load_env "${ENV_FILE}"

# Defaults for global settings.
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-180}"
VLLM_PREFIX_CACHING_HASH_ALGO="${VLLM_PREFIX_CACHING_HASH_ALGO:-xxhash}"

# =============================================================================
# Instance Management
# =============================================================================

start_instance() {
    local role="$1"

    # Read per-role configuration via helper.
    local model port device gpu_memory_mb attention_backend extra_args
    model="$(vllm_role_config "${role}" MODEL)"
    port="$(vllm_role_config "${role}" PORT)"
    device="$(vllm_role_config "${role}" DEVICE)"
    gpu_memory_mb="$(vllm_role_config "${role}" GPU_MEMORY_MB)"
    attention_backend="$(vllm_role_config "${role}" ATTENTION_BACKEND)"
    extra_args="$(vllm_role_config "${role}" EXTRA_ARGS)"

    local pid_file log_file
    pid_file="$(vllm_pid_file "${role}" "${PLATFORM_DIR}")"
    log_file="$(vllm_log_file "${role}" "${PLATFORM_DIR}")"

    # Validate required fields.
    local role_upper
    role_upper="$(vllm_role_upper "${role}")"
    if [[ -z "${model}" ]]; then
        die "Missing VLLM_${role_upper}_MODEL in .env"
    fi
    if [[ -z "${port}" ]]; then
        die "Missing VLLM_${role_upper}_PORT in .env"
    fi
    if [[ -z "${device}" ]]; then
        die "Missing VLLM_${role_upper}_DEVICE in .env"
    fi

    # Check if already running.
    if vllm_is_running "${role}" "${PLATFORM_DIR}"; then
        local pid
        pid="$(vllm_read_pid "${role}" "${PLATFORM_DIR}")"
        warn "vLLM ${role} already running (PID: ${pid}). Skipping."
        return 0
    fi

    # Clean up stale PID if present but process dead.
    local existing_pid
    existing_pid="$(vllm_read_pid "${role}" "${PLATFORM_DIR}")"
    if [[ -n "${existing_pid}" ]]; then
        vllm_cleanup_stale_pid "${role}" "${PLATFORM_DIR}"
    fi

    # Build command arguments.
    # vLLM has no --device CLI flag. Device selection is controlled via the
    # VLLM_TARGET_DEVICE environment variable (defaults to "cuda"). For
    # non-GPU roles (cpu), we set this per-process at launch time.
    local -a cmd_args=(
        vllm serve "${model}"
        --host "${VLLM_HOST}"
        --port "${port}"
        --enable-prefix-caching
        --prefix-caching-hash-algo "${VLLM_PREFIX_CACHING_HASH_ALGO}"
    )

    # Attention backend override: allows per-role selection of attention backend.
    # When set, forces a specific backend instead of vLLM's auto-selection.
    # Valid values: ROCM_AITER_FA, ROCM_AITER_UNIFIED_ATTN, TRITON_ATTN, etc.
    if [[ -n "${attention_backend}" ]]; then
        cmd_args+=(--override-attention-backend "${attention_backend}")
        info "${role}: attention backend override: ${attention_backend}"
    fi

    # GPU memory: convert MB to utilization fraction.
    if [[ -n "${gpu_memory_mb}" ]]; then
        local total_mb utilization
        total_mb="$(vllm_gtt_total_mb)"
        utilization="$(vllm_mb_to_utilization "${gpu_memory_mb}" "${total_mb}")"
        cmd_args+=(--gpu-memory-utilization "${utilization}")
        info "${role}: GPU memory ${gpu_memory_mb}MB / ${total_mb}MB = ${utilization}"
    fi

    # Append extra args (word-split intentionally).
    # shellcheck disable=SC2206
    cmd_args+=(${extra_args})

    info "Starting vLLM ${role}: ${model} on ${device}:${port}"
    info "Log file: ${log_file}"

    # Launch in background with per-process VLLM_TARGET_DEVICE.
    VLLM_TARGET_DEVICE="${device}" nohup "${cmd_args[@]}" > "${log_file}" 2>&1 &

    local instance_pid=$!
    echo "${instance_pid}" > "${pid_file}"

    # Health check loop.
    info "Waiting for ${role} health check (timeout: ${VLLM_STARTUP_TIMEOUT}s)..."

    if vllm_poll_health "${VLLM_HOST}" "${port}" "${VLLM_STARTUP_TIMEOUT}" "${instance_pid}"; then
        success "vLLM ${role} ready (PID: ${instance_pid}, port: ${port})"

        # Log which attention backend was actually selected (parse vLLM log).
        local backend_line
        backend_line="$(grep -oP 'Using \K\S+ out of potential backends' "${log_file}" 2>/dev/null | head -1)"
        if [[ -n "${backend_line}" ]]; then
            info "${role}: ${backend_line}"
        fi
        return 0
    fi

    # Failed: check if process died or timed out.
    if ! kill -0 "${instance_pid}" 2>/dev/null; then
        error "vLLM ${role} (PID ${instance_pid}) died during startup. Last 30 lines:"
    else
        error "vLLM ${role} did not become healthy within ${VLLM_STARTUP_TIMEOUT}s. Last 30 lines:"
    fi
    tail -30 "${log_file}" >&2
    rm -f "${pid_file}"
    return 1
}

# =============================================================================
# Main
# =============================================================================

main() {
    section "vLLM Inference Server"

    require_commands vllm curl bc

    vllm_require_roles

    # Log optimization state for debugging.
    vllm_log_optimization_state

    local failed=0

    for role in ${VLLM_ROLES}; do
        if ! start_instance "${role}"; then
            error "Failed to start ${role}"
            failed=$((failed + 1))
        fi
    done

    if [[ "${failed}" -gt 0 ]]; then
        die "${failed} instance(s) failed to start."
    fi

    success "All vLLM instances running."
    info "Stop with: scripts/vllm-stop.sh"
}

main "$@"
