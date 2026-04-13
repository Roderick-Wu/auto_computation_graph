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
        echo "Running reject-traces locally for ${experiment} (${n_formats} formats) on ${MODEL_NAME}"
        bash reject_traces.sh "$experiment" "$MODEL_NAME"
    else
        echo "Submitting reject-traces job for ${experiment} (${n_formats} formats) on ${MODEL_NAME}"
        sbatch reject_traces.sh "$experiment" "$MODEL_NAME"
    fi
done < <(python list_all_experiments.py)

if [[ "$RUN_MODE" == "local" ]]; then
    echo "All reject-traces runs completed locally."
else
    echo "All reject-traces jobs submitted."
fi