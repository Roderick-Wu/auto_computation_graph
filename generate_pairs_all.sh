#!/bin/bash
set -euo pipefail

while IFS=$'\t' read -r experiment n_formats
do
    [ -z "$experiment" ] && continue
    echo "Submitting pair-generation job for ${experiment} (${n_formats} formats)"
    bash generate_pairs.sh "$experiment" "Qwen2.5-32B"
done < <(python list_all_experiments.py)

echo "All pair generation jobs submitted."