#!/bin/bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen2.5-32B}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-16}"
TOKEN_POSITIONS_TO_PATCH="${TOKEN_POSITIONS_TO_PATCH:-all}"

while read -r experiment n_formats
do
    [ -z "$experiment" ] && continue
    echo "Submitting counterfactual token-swap job for ${experiment} (${n_formats} formats) on ${MODEL_NAME}"
    sbatch --export=ALL,MODEL_NAME="$MODEL_NAME",PATCH_BATCH_SIZE="$PATCH_BATCH_SIZE",TOKEN_POSITIONS_TO_PATCH="$TOKEN_POSITIONS_TO_PATCH" patch_graph.sh "$experiment" "$MODEL_NAME"
done < <(python list_all_experiments.py)

echo "All counterfactual token-swap jobs submitted."
