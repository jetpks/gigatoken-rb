#!/usr/bin/env bash
# Analyze a trace produced by profile.sh.
#
# Usage:
#   ./analyze.sh traces/samply_10000mb.json.gz [path/to/encode_st-binary]
#       -> <trace>_analysis/{top_functions.txt, collapsed.folded, hot_lines.txt}
#   ./analyze.sh traces/counters_10000mb.trace [--window START_S END_S] [--pcore-only]
#       -> PMU bottleneck summary on stdout
set -euo pipefail

TRACE="$1"; shift || true
DIR="$(cd "$(dirname "$0")" && pwd)"

case "$TRACE" in
    *.json.gz)
        BIN="${1:-}"
        if [ -z "$BIN" ]; then
            BIN=$(cat "${TRACE%.json.gz}.bin" 2>/dev/null || true)
        fi
        [ -n "$BIN" ] || { echo "usage: analyze.sh TRACE.json.gz BINARY"; exit 1; }
        python3 "$DIR/analyze.py" "$TRACE" --bin "$BIN" --hot-regions 5
        ;;
    *.trace)
        python3 "$DIR/pmu_summary.py" "$TRACE" "$@"
        ;;
    *)
        echo "unrecognized trace type: $TRACE"; exit 1
        ;;
esac
