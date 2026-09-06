#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV
# Fail before calibration or output-directory creation on login/CPU nodes,
# non-H100 GPUs, and restricted H100 partitions such as small MIG slices.
uv run --no-sync python -m looped_transformer_comparison.cli check --require-h100
exec uv run --no-sync python -m looped_transformer_comparison.cli budget "$@"
