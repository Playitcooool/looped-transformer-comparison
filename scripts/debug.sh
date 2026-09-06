#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
exec uv run --no-sync python -m looped_transformer_comparison.cli train "$@"
