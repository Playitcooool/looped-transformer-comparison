#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV
exec uv run --no-sync looped-transformer-comparison generate "$@"
