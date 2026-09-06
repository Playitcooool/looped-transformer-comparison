#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }
unset VIRTUAL_ENV
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
uv sync --locked
uv run --no-sync python -m looped_transformer_comparison.cli check
