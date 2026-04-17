#!/bin/bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen2.5-32B}"
LAYER_STRIDE="${LAYER_STRIDE:-8}"

while read -r experiment n_formats
do
    [ -z "$experiment" ] && continue
    echo "Submitting patching job for ${experiment} (${n_formats} formats) on ${MODEL_NAME} (layer_stride=${LAYER_STRIDE})"
    sbatch --export=ALL,MODEL_NAME="$MODEL_NAME",LAYER_STRIDE="$LAYER_STRIDE" patch_graph.sh "$experiment" "$MODEL_NAME"
done < <(python list_all_experiments.py)

echo "All patching jobs submitted."
