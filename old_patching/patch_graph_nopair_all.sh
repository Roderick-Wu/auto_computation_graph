#!/bin/bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen2.5-32B}"
LAYER_STRIDE="${LAYER_STRIDE:-8}"
PATCH_SCOPE="${PATCH_SCOPE:-token_all_layers}"

while read -r experiment n_formats
do
    [ -z "$experiment" ] && continue
    echo "Submitting no-pair patching job for ${experiment} (${n_formats} formats) on ${MODEL_NAME} (layer_stride=${LAYER_STRIDE}, patch_scope=${PATCH_SCOPE})"
    sbatch --export=ALL,MODEL_NAME="$MODEL_NAME",LAYER_STRIDE="$LAYER_STRIDE",PATCH_SCOPE="$PATCH_SCOPE" patch_graph_nopair.sh "$experiment" "$MODEL_NAME"
done < <(python list_all_experiments.py)

echo "All no-pair patching jobs submitted."
