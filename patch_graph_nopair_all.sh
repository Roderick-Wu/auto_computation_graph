#!/bin/bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen2.5-32B}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-16}"
TOKEN_POSITIONS_TO_PATCH="${TOKEN_POSITIONS_TO_PATCH:-all}"
NOISE_MODE="${NOISE_MODE:-gaussian}"
NOISE_SCALE="${NOISE_SCALE:-1.0}"
NOISE_SEED="${NOISE_SEED:-0}"
NOISE_SAMPLES_PER_TOKEN="${NOISE_SAMPLES_PER_TOKEN:-10}"

while read -r experiment n_formats
do
    [ -z "$experiment" ] && continue
    echo "Submitting no-pair token-noise job for ${experiment} (${n_formats} formats) on ${MODEL_NAME}"
    sbatch --export=ALL,MODEL_NAME="$MODEL_NAME",PATCH_BATCH_SIZE="$PATCH_BATCH_SIZE",TOKEN_POSITIONS_TO_PATCH="$TOKEN_POSITIONS_TO_PATCH",NOISE_MODE="$NOISE_MODE",NOISE_SCALE="$NOISE_SCALE",NOISE_SEED="$NOISE_SEED",NOISE_SAMPLES_PER_TOKEN="$NOISE_SAMPLES_PER_TOKEN" patch_graph_nopair.sh "$experiment" "$MODEL_NAME"
done < <(python list_all_experiments.py)

echo "All no-pair token-noise jobs submitted."
