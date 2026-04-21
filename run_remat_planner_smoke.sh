#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "Building PCP..."
nix build

echo "Running rematerialization planner smoke test..."
./result/bin/remat_planner_smoke
