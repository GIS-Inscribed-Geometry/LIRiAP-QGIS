#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "=== Building lir_solver (release) ==="
pip install "maturin>=1.4,<2.0" -q
maturin build --release --interpreter python3 --out dist/
pip install dist/lir_solver-*.whl --force-reinstall
echo "=== Done. Test: python python/benchmark.py ==="
