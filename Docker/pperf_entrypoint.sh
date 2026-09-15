#!/usr/bin/env bash
set -euo pipefail

readonly TRACEFS_SOURCE=/sys/kernel/tracing
readonly BPFTRACE_TRACEFS=/sys/kernel/debug/tracing

# bpftrace 0.14 reads tracepoints only from /sys/kernel/debug/tracing.
# Expose the host tracefs there without modifying the host mount namespace.
if [[ -r "${TRACEFS_SOURCE}/available_events" &&
      ! -r "${BPFTRACE_TRACEFS}/available_events" ]]; then
    mount -t tmpfs -o mode=755 tmpfs /sys/kernel/debug
    mkdir -p "${BPFTRACE_TRACEFS}"
    mount --bind "${TRACEFS_SOURCE}" "${BPFTRACE_TRACEFS}"
fi

exec /opt/nvidia/nvidia_entrypoint.sh "$@"
