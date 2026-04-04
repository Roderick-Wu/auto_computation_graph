#!/bin/bash
# Batch node skipping: run node skipping tests on all registered experiments

set -euo pipefail

MODEL_NAME="${1:-Qwen2.5-72B}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/auto_computation_graph"

echo "Running node skipping tests for all experiments on $MODEL_NAME"

while read -r experiment n_formats; do
    echo "===== $experiment ====="
    bash skip_nodes.sh "$MODEL_NAME" "$experiment" "" 5
done < <(python list_all_experiments.py)

echo "All node skipping tests complete."
