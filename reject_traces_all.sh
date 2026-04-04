#!/bin/bash
set -euo pipefail

while IFS=$'\t' read -r experiment n_formats
do
    [ -z "$experiment" ] && continue
    echo "Submitting reject-traces job for ${experiment} (${n_formats} formats)"
    bash reject_traces.sh "$experiment" "Qwen2.5-32B"
done < <(python list_all_experiments.py)

echo "All reject-traces jobs submitted."