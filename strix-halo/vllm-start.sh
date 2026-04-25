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

# Source shared helpers (logging, section headers, prerequisite checks).
# shellcheck source=common.sh
source "${_SCRIPT_DIR}/common.sh"

# shellcheck source=vllm-env.sh disable=SC1091
source "${_SCRIPT_DIR}/vllm-env.sh"

# shellcheck source=vllm-runtime-helpers.sh
source "${_SCRIPT_DIR}/vllm-runtime-helpers.sh"

PLATFORM_DIR="${_SCRIPT_DIR}"
ENV_FILE="${_SCRIPT_DIR}/.env"

unset _SCRIPT_REAL_PATH _SCRIPT_DIR

# Load .env configuration.
vllm_load_env "${ENV_FILE}"

# Defaults for global settings.
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-180}"
VLLM_PREFIX_CACHING_HASH_ALGO="${VLLM_PREFIX_CACHING_HASH_ALGO:-xxhash}"
VLLM_STARTUP_ERROR_TAIL_LINES="${VLLM_STARTUP_ERROR_TAIL_LINES:-120}"
VLLM_MAX_GPU_MEMORY_UTILIZATION="${VLLM_MAX_GPU_MEMORY_UTILIZATION:-0.98}"

# Print a startup failure summary from a vLLM log file.
#
# Args:
#   log_file - Path to vLLM instance log file
vllm_print_startup_failure_details() {
    local log_file="$1"
    local traceback_context_lines="${VLLM_STARTUP_TRACEBACK_CONTEXT_LINES:-40}"

    if [[ ! -f "${log_file}" ]]; then
        error "No startup log found at ${log_file}"
        return
    fi

    # vLLM often emits a generic RuntimeError at the end of startup failure.
    # Printing where the first traceback starts helps locate root cause quickly.
    local first_traceback_line
    first_traceback_line="$(grep -n -m1 "Traceback (most recent call last)" "${log_file}" \
        | cut -d: -f1 || true)"
    if [[ -n "${first_traceback_line}" ]]; then
        error "First traceback starts at line ${first_traceback_line} in ${log_file}"

        # Print focused context around the first traceback, because vLLM often
        # reports the true engine/core failure immediately before it.
        local context_start context_end
        context_start=$(( first_traceback_line - traceback_context_lines ))
        if [[ "${context_start}" -lt 1 ]]; then
            context_start=1
        fi
        context_end=$(( first_traceback_line + traceback_context_lines ))
        error "Context around first traceback (lines ${context_start}-${context_end}):"
        sed -n "${context_start},${context_end}p" "${log_file}" >&2
    fi

    error "Last ${VLLM_STARTUP_ERROR_TAIL_LINES} lines from ${log_file}:"
    tail -"${VLLM_STARTUP_ERROR_TAIL_LINES}" "${log_file}" >&2
}

# =============================================================================
# Instance Management
# =============================================================================


# Detect known torch.compile duplicate-pattern crash in ROCm AITER RMSNorm fusion.
#
# Args:
#   log_file - Path to vLLM instance log file
# Returns:
#   0 if duplicate-pattern signature found, 1 otherwise
vllm_is_aiter_rmsnorm_duplicate_pattern_failure() {
    local log_file="$1"

    [[ -f "${log_file}" ]] || return 1

    grep -q "rocm_aiter_fusion.py" "${log_file}" \
        && grep -q "check_and_add_duplicate_pattern" "${log_file}"
}

# Print targeted diagnostics for the duplicate-pattern startup crash.
#
# This focuses on root-cause indicators instead of only applying fallbacks:
#   - Installed vLLM / torch / triton versions
#   - rocm_aiter_fusion.py path from the active Python environment
#   - Whether skip_duplicates=True is present on register_replacement calls
#
# Args:
#   role - Logical instance role (main, etc.) for log prefixing
vllm_print_duplicate_pattern_diagnostics() {
    local role="$1"

    info "${role}: collecting duplicate-pattern diagnostics (versions + patch state)"
    python - <<'PY'
import importlib.util
import os
import re
import sys

def _safe_import(name):
    try:
        mod = __import__(name)
        return mod, None
    except Exception as exc:  # pragma: no cover - diagnostics only
        return None, exc

vllm, vllm_err = _safe_import("vllm")
torch, torch_err = _safe_import("torch")
triton, triton_err = _safe_import("triton")

print("  diag  Python executable:", sys.executable)
print("  diag  vLLM version:      ", getattr(vllm, "__version__", f"<import failed: {vllm_err}>"))
print("  diag  torch version:     ", getattr(torch, "__version__", f"<import failed: {torch_err}>"))
print("  diag  triton version:    ", getattr(triton, "__version__", f"<import failed: {triton_err}>"))

target = None
if vllm is not None:
    spec = importlib.util.find_spec("vllm.compilation.passes.fusion.rocm_aiter_fusion")
    if spec is not None:
        target = spec.origin

if not target or not os.path.isfile(target):
    print("  diag  rocm_aiter_fusion: <not found in active environment>")
    raise SystemExit(0)

print("  diag  rocm_aiter_fusion: ", target)
try:
    text = open(target, "r", encoding="utf-8").read()
except Exception as exc:  # pragma: no cover - diagnostics only
    print(f"  diag  patch check:       <failed to read file: {exc}>")
    raise SystemExit(0)

try:
    register_calls = len(re.findall(r"pm\.register_replacement\(", text))
    skip_dupe = len(
        re.findall(
            r"pm\.register_replacement\([^\n]*skip_duplicates\s*=\s*True",
            text,
        )
    )
except re.error as exc:  # pragma: no cover - diagnostics only
    print(f"  diag  regex check:       <failed: {exc}>")
    raise SystemExit(0)

print("  diag  register calls:    ", register_calls)
print("  diag  skip_duplicates=:  ", skip_dupe)

if register_calls and skip_dupe == 0:
    print("  diag  likely root cause: active wheel is missing the skip_duplicates patch")
    print("  diag  action: rebuild/reinstall vLLM so rocm_aiter_fusion.py includes skip_duplicates=True")
elif register_calls and skip_dupe < register_calls:
    print("  diag  likely root cause: partial patch application in rocm_aiter_fusion.py")
else:
    print("  diag  patch state:       skip_duplicates appears present")
PY
}

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
        if [[ "$(echo "${utilization} > ${VLLM_MAX_GPU_MEMORY_UTILIZATION}" | bc)" -eq 1 ]]; then
            warn "${role}: requested ${gpu_memory_mb}MB exceeds detected ${total_mb}MB; capping --gpu-memory-utilization to ${VLLM_MAX_GPU_MEMORY_UTILIZATION}"
            utilization="${VLLM_MAX_GPU_MEMORY_UTILIZATION}"
        fi
        cmd_args+=(--gpu-memory-utilization "${utilization}")
        info "${role}: GPU memory ${gpu_memory_mb}MB / ${total_mb}MB = ${utilization}"
    fi

    # Append extra args (word-split intentionally).
    # shellcheck disable=SC2206
    cmd_args+=(${extra_args})

    # Preferred switch: explicit enable for one-time retry fallback.
    # Legacy compatibility: if VLLM_DISABLE_AITER_RMSNORM_ON_DUP_PATTERN=1 is
    # set, treat that as enabled as well.
    local enable_rmsnorm_retry="${VLLM_ENABLE_AITER_RMSNORM_DUP_PATTERN_RETRY:-0}"
    if [[ "${enable_rmsnorm_retry}" != "1" ]] \
        && [[ "${VLLM_DISABLE_AITER_RMSNORM_ON_DUP_PATTERN:-0}" == "1" ]]; then
        enable_rmsnorm_retry="1"
    fi
    local launch_with_rmsnorm_disabled=0
    local attempt

    for attempt in 1 2; do
        info "Starting vLLM ${role}: ${model} on ${device}:${port} (attempt ${attempt}/2)"
        info "Log file: ${log_file}"

        # Launch in background with per-process VLLM_TARGET_DEVICE.
        local -a launch_env=(
            VLLM_TARGET_DEVICE="${device}"
        )
        if [[ "${launch_with_rmsnorm_disabled}" -eq 1 ]]; then
            launch_env+=(VLLM_ROCM_USE_AITER_RMSNORM=0)
        fi
        env "${launch_env[@]}" nohup "${cmd_args[@]}" > "${log_file}" 2>&1 &

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
            error "vLLM ${role} (PID ${instance_pid}) died during startup."
        else
            error "vLLM ${role} did not become healthy within ${VLLM_STARTUP_TIMEOUT}s."
        fi

        # Automatic one-time fallback for known duplicate pattern crash in
        # RocmAiterRMSNormQuantFusionPass.
        if [[ "${VLLM_ROCM_USE_AITER_RMSNORM:-0}" == "1" ]] \
            && vllm_is_aiter_rmsnorm_duplicate_pattern_failure "${log_file}"; then
            vllm_print_duplicate_pattern_diagnostics "${role}"

            if [[ "${enable_rmsnorm_retry}" == "1" ]] \
                && [[ "${launch_with_rmsnorm_disabled}" -eq 0 ]]; then
                launch_with_rmsnorm_disabled=1
                warn "${role}: detected AITER RMSNorm duplicate-pattern startup crash; retrying with VLLM_ROCM_USE_AITER_RMSNORM=0"
                rm -f "${pid_file}"
                continue
            fi
            warn "${role}: detected AITER RMSNorm duplicate-pattern startup crash"
            warn "${role}: retry fallback is disabled (set VLLM_ENABLE_AITER_RMSNORM_DUP_PATTERN_RETRY=1 to enable one-time retry)"
        fi

        vllm_print_startup_failure_details "${log_file}"
        rm -f "${pid_file}"
        return 1
    done

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
