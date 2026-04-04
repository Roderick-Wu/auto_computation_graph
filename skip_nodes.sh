#!/bin/bash
# Test node skipping: can we bypass intermediate values?

set -euo pipefail

MODEL_NAME="${1:-Qwen2.5-72B}"
EXPERIMENT="${2:-velocity}"
GRAPH_DIR="${3:-}"
MAX_PAIRS="${4:-5}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/auto_computation_graph"

if [[ -z "$GRAPH_DIR" ]]; then
    SCRATCH_ROOT="$HOME/links/scratch"
    [[ ! -d "$SCRATCH_ROOT" ]] && SCRATCH_ROOT="$HOME/scratch"
    GRAPH_DIR="$SCRATCH_ROOT/traces/$MODEL_NAME/$EXPERIMENT/patch_runs"
fi

echo "Testing node skipping for: $EXPERIMENT on $MODEL_NAME"
echo "Graph directory: $GRAPH_DIR"

python intervene_skip_nodes.py \
    --model-name "$MODEL_NAME" \
    --experiment "$EXPERIMENT" \
    --graph-dir "$GRAPH_DIR" \
    --max-pairs "$MAX_PAIRS" \
    --max-skip-depths 3 \
    --device cuda \
    --dtype float16

echo "Skipping tests complete. Check node_skipping_results.json in experiment logs."
