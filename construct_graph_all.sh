#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../workspace_paths.sh"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-32B}"
GRAPH_VARIANT="${GRAPH_VARIANT:-pair}"
USE_SLURM="${USE_SLURM:-0}"

while read -r experiment n_formats
do
    [ -z "$experiment" ] && continue
    traces_dir="$WRODERI_SCRATCH_ROOT/traces/$MODEL_NAME/$experiment"
    case "$GRAPH_VARIANT" in
        pair|regular)
            patch_dir="$traces_dir/patch_runs"
            ;;
        nopair|solo|noise)
            patch_dir="$traces_dir/patch_solo"
            ;;
        *)
            echo "ERROR: Unknown graph variant: $GRAPH_VARIANT"
            exit 1
            ;;
    esac

    if [ ! -f "$patch_dir/patching_summary.json" ]; then
        echo "Skipping graph construction for ${experiment}: missing $patch_dir/patching_summary.json"
        continue
    fi

    if [ "$USE_SLURM" = "1" ]; then
        echo "Submitting graph construction job for ${experiment} (${n_formats} formats), variant=${GRAPH_VARIANT}"
        sbatch --export=ALL,MODEL_NAME="$MODEL_NAME",GRAPH_VARIANT="$GRAPH_VARIANT",USE_SLURM="$USE_SLURM" construct_graph.sh "$experiment" "$MODEL_NAME" "$GRAPH_VARIANT"
    else
        echo "Running graph construction for ${experiment} (${n_formats} formats), variant=${GRAPH_VARIANT}"
        bash construct_graph.sh "$experiment" "$MODEL_NAME" "$GRAPH_VARIANT"
    fi
done < <(python list_all_experiments.py)

if [ "$USE_SLURM" = "1" ]; then
    echo "Graph construction submission sweep complete."
else
    echo "Graph construction local sweep complete."
fi
