#!/usr/bin/env bash
# GPU sweep at the standard fleet sizes, then tabulate everything in benchmark/data/.
set -euo pipefail
cd "$(dirname "$0")/.."

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50

uv run python benchmark/main.py --device gpu --worlds 64,1024,16384,65536,262144
uv run python benchmark/compare.py benchmark/data/*.csv
