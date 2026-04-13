#!/bin/bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen2.5-72B}"
RUN_MODE="${RUN_MODE:-local}"

if [[ "$RUN_MODE" != "local" && "$RUN_MODE" != "slurm" ]]; then
    echo "ERROR: RUN_MODE must be 'local' or 'slurm' (got '$RUN_MODE')"
    exit 1
fi

while IFS=$'\t' read -r experiment n_formats
do
    [ -z "$experiment" ] && continue
    if [[ "$RUN_MODE" == "local" ]]; then
        echo "Running pair-generation locally for ${experiment} (${n_formats} formats) on ${MODEL_NAME}"
        bash generate_pairs.sh "$experiment" "$MODEL_NAME"
    else
        echo "Submitting pair-generation job for ${experiment} (${n_formats} formats) on ${MODEL_NAME}"
        sbatch generate_pairs.sh "$experiment" "$MODEL_NAME"
    fi
done < <(python list_all_experiments.py)

if [[ "$RUN_MODE" == "local" ]]; then
    echo "All pair-generation runs completed locally."
else
    echo "All pair-generation jobs submitted."
fi