#!/usr/bin/env bash
# Copyright 2026 Blackcat Informatics Inc.
# SPDX-License-Identifier: MIT
#
# vllm-stop.sh - Stop all vLLM inference instances
#
# Sends SIGTERM to each vLLM process listed in VLLM_ROLES, waits for
# graceful shutdown, and falls back to SIGKILL after a timeout.
#
# Usage:
#   scripts/vllm-stop.sh

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

# Load .env for VLLM_ROLES.
vllm_load_env "${ENV_FILE}"

# Seconds to wait for graceful shutdown before sending SIGKILL.
SHUTDOWN_TIMEOUT=30

# =============================================================================
# Instance Management
# =============================================================================

stop_instance() {
    local role="$1"
    local pid_file
    pid_file="$(vllm_pid_file "${role}" "${PLATFORM_DIR}")"

    local pid
    pid="$(vllm_read_pid "${role}" "${PLATFORM_DIR}")"

    if [[ -z "${pid}" ]]; then
        warn "No PID file for ${role}. May not be running."
        return 0
    fi

    if ! kill -0 "${pid}" 2>/dev/null; then
        vllm_cleanup_stale_pid "${role}" "${PLATFORM_DIR}"
        return 0
    fi

    info "Sending SIGTERM to vLLM ${role} (PID ${pid})..."
    kill -TERM "${pid}"

    # Wait for graceful shutdown.
    local waited=0
    while kill -0 "${pid}" 2>/dev/null && [[ "${waited}" -lt "${SHUTDOWN_TIMEOUT}" ]]; do
        sleep 1
        waited=$((waited + 1))
    done

    if kill -0 "${pid}" 2>/dev/null; then
        warn "vLLM ${role} did not exit within ${SHUTDOWN_TIMEOUT}s. Sending SIGKILL."
        kill -KILL "${pid}" 2>/dev/null || true
        sleep 1
    fi

    rm -f "${pid_file}"
    success "vLLM ${role} stopped."
}

# =============================================================================
# Main
# =============================================================================

main() {
    section "Stopping vLLM Inference Server"

    vllm_require_roles

    for role in ${VLLM_ROLES}; do
        stop_instance "${role}"
    done

    success "All vLLM instances stopped."
}

main "$@"
