#!/bin/bash
# Batch validation: run causal structure tests on all registered experiments

set -euo pipefail

MODEL_NAME="${1:-Qwen2.5-32B}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/../workspace_paths.sh"

cd "$WRODERI_PROJECT_ROOT/auto_computation_graph"

echo "Running causal structure validation for all experiments on $MODEL_NAME"

while read -r experiment n_formats; do
    echo "===== $experiment ====="
    sbatch validate_causal_structure.sh "$MODEL_NAME" "$experiment" "" 5
done < <(python list_all_experiments.py)

echo "All validation runs complete."
