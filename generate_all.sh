#!/bin/bash
set -euo pipefail

# Number of examples to generate per prompt format (override with env var if needed).
SAMPLES_PER_FORMAT=${SAMPLES_PER_FORMAT:-10}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}

if [ "$SAMPLES_PER_FORMAT" -lt 1 ]; then
    echo "ERROR: SAMPLES_PER_FORMAT must be >= 1"
    exit 1
fi

MODEL_NAME=${MODEL_NAME:-Qwen2.5-72B}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

# Pull all available generator names + their prompt format counts from prompts.py.
while IFS=$'\t' read -r experiment n_formats
do
    [ -z "$experiment" ] && continue
    N_PROMPTS=$((n_formats * SAMPLES_PER_FORMAT))
    echo "Submitting ${experiment}: ${n_formats} formats x ${SAMPLES_PER_FORMAT} samples/format = ${N_PROMPTS} prompts on ${GPUS_PER_NODE}x H100"
    sbatch --export=ALL,MAX_NEW_TOKENS="$MAX_NEW_TOKENS" --gpus-per-node="h100:${GPUS_PER_NODE}" generate.sh "$experiment" "$MODEL_NAME" "$N_PROMPTS"
done < <(python list_all_experiments.py)