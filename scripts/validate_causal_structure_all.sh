#!/bin/bash
# Batch validation: run causal structure tests on all registered experiments

set -euo pipefail

MODEL_NAME="${1:-Qwen2.5-32B}"
MAX_PAIRS="${2:-0}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/../workspace_paths.sh"

cd "$WRODERI_REPO_ROOT"

echo "Running causal structure validation for all experiments on $MODEL_NAME"

while read -r experiment n_formats; do
    echo "===== $experiment ====="
    sbatch "$SCRIPT_DIR/validate_causal_structure.sh" "$MODEL_NAME" "$experiment" "" "$MAX_PAIRS"
done < <(python "$WRODERI_REPO_ROOT/src/list_all_experiments.py")

echo "All validation runs complete."
