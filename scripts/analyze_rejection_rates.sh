#!/bin/bash
# Analyze acceptance rates across all experiments

set -euo pipefail

MODEL_NAME="${1:-Qwen2.5-72B}"
LOW_THRESHOLD="${2:-0.50}"
CONCERNING_THRESHOLD="${3:-0.70}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../workspace_paths.sh"

cd "$WRODERI_REPO_ROOT"

echo "Analyzing rejection rates for: $MODEL_NAME"
echo "Low threshold: $LOW_THRESHOLD"
echo "Concerning threshold: $CONCERNING_THRESHOLD"
echo

python "$WRODERI_REPO_ROOT/src/analyze_rejection_rates.py" \
    --model-name "$MODEL_NAME" \
    --low-threshold "$LOW_THRESHOLD" \
    --concerning-threshold "$CONCERNING_THRESHOLD"
