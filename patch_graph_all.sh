#!/bin/bash
set -euo pipefail

while read -r experiment n_formats
do
    [ -z "$experiment" ] && continue
    echo "Submitting patching job for ${experiment} (${n_formats} formats)"
    sbatch patch_graph.sh "$experiment" "Qwen2.5-32B"
done < <(python list_all_experiments.py)

echo "All patching jobs submitted."