#!/bin/bash
# Analyze acceptance rates across all experiments

set -euo pipefail

MODEL_NAME="${1:-Qwen2.5-72B}"
LOW_THRESHOLD="${2:-0.50}"
CONCERNING_THRESHOLD="${3:-0.70}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/auto_computation_graph"

echo "Analyzing rejection rates for: $MODEL_NAME"
echo "Low threshold: $LOW_THRESHOLD"
echo "Concerning threshold: $CONCERNING_THRESHOLD"
echo

python analyze_rejection_rates.py \
    --model-name "$MODEL_NAME" \
    --low-threshold "$LOW_THRESHOLD" \
    --concerning-threshold "$CONCERNING_THRESHOLD"
