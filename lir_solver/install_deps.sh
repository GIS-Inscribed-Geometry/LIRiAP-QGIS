#!/usr/bin/env bash
set -euo pipefail
if ! command -v rustc &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
fi
rustup override set stable
pip install "maturin>=1.4,<2.0" shapely numpy -q
cd "$(dirname "$0")"
bash build.sh
