#!/usr/bin/env bash
# DEPRECATED fail-closed shim: stale invocations must not write into the
# shared results/ tree. All benchmarking is delegated to rebench.sh, which
# requires explicit MODEL_DIR and LLAMA_SERVER and fails closed otherwise.
set -euo pipefail
echo "bench/run-all.sh is deprecated; delegating to bench/rebench.sh" >&2
exec bash "$(dirname "$0")/rebench.sh" "$@"
