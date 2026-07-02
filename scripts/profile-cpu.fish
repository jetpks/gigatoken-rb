#!/usr/bin/env fish
# Record a CPU profile of a bench with Instruments' "CPU Profiler" template,
# via cargo-instruments (which demangles Rust v0 symbols itself).
#
# Usage: ./scripts/profile-cpu.fish [bench-name]      (default: encode_st)
#   ENCODE_MB=<n>   cap the input (default 500); passed through to the bench
#
# The trace is written to the repo root as <bench>_cpu.trace and opened in
# Instruments.app. Must be run from a real terminal: cargo-instruments needs a
# controlling TTY (it hands the target a /dev/stdout that is invalid otherwise).
#
# Requires: cargo install cargo-instruments

set -l bench (test -n "$argv[1]"; and echo $argv[1]; or echo encode_st)
set -l root (git rev-parse --show-toplevel)
set -l trace "$root/$bench"_cpu.trace

# Default the input cap unless the caller already set one.
set -q ENCODE_MB; or set -x ENCODE_MB 500

if not type -q cargo-instruments
    echo "cargo-instruments not found. Install with: cargo install cargo-instruments" >&2
    exit 1
end

# Fresh single-run trace (cargo-instruments would otherwise append a Run).
rm -rf "$trace"

# CARGO_INCREMENTAL=0: incremental codegen units can cache symbol names from a
# prior mangling setting, leaving stale symbols in the build.
CARGO_INCREMENTAL=0 cargo instruments \
    -t cpu \
    --bench $bench \
    --profile profiling \
    -o "$trace"; or exit $status

echo "Trace saved to $trace (ENCODE_MB=$ENCODE_MB)"
