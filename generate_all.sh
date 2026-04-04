#!/bin/bash
set -euo pipefail

# Number of examples to generate per prompt format (override with env var if needed).
SAMPLES_PER_FORMAT=${SAMPLES_PER_FORMAT:-10}

if [ "$SAMPLES_PER_FORMAT" -lt 1 ]; then
    echo "ERROR: SAMPLES_PER_FORMAT must be >= 1"
    exit 1
fi

# Pull all available generator names + their prompt format counts from prompts.py.
while IFS=$'\t' read -r experiment n_formats
do
    [ -z "$experiment" ] && continue
    N_PROMPTS=$((n_formats * SAMPLES_PER_FORMAT))
    echo "Submitting ${experiment}: ${n_formats} formats x ${SAMPLES_PER_FORMAT} samples/format = ${N_PROMPTS} prompts"
    sbatch generate.sh "$experiment" "$N_PROMPTS"
done < <(python list_all_experiments.py)