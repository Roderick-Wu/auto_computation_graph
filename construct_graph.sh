#!/bin/bash
#SBATCH --job-name=construct_graph
#SBATCH --time=0-02:00:00 # D-HH:MM
#SBATCH --account=def-zhijing
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Load required modules
module load python/3.11.5 scipy-stack/2023b arrow/21.0.0

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/../workspace_paths.sh"

# Default experiment (e.g., "velocity")
EXPERIMENT=${1:-velocity}
MODEL_NAME=${2:-Qwen2.5-32B}
GRAPH_VARIANT=${3:-${GRAPH_VARIANT:-pair}}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/../workspace_paths.sh"

# Input/output paths
TRACES_DIR="$WRODERI_SCRATCH_ROOT/traces/${MODEL_NAME}/${EXPERIMENT}"
case "$GRAPH_VARIANT" in
    pair|regular)
        PATCH_MATRIX_DIR="${TRACES_DIR}/patch_runs"
        OUTPUT_DIR="${TRACES_DIR}/graphs"
        IS_NOPAIR_VARIANT=0
        ;;
    nopair|solo|noise)
        PATCH_MATRIX_DIR="${TRACES_DIR}/patch_solo"
        OUTPUT_DIR="${TRACES_DIR}/graphs_nopair"
        IS_NOPAIR_VARIANT=1
        ;;
    *)
        echo "ERROR: Unknown graph variant: $GRAPH_VARIANT"
        echo "Expected one of: pair, regular, nopair, solo, noise"
        exit 1
        ;;
esac

if [ ! -d "$PATCH_MATRIX_DIR" ]; then
    echo "ERROR: Patch matrix directory not found: $PATCH_MATRIX_DIR"
    echo "Please run the corresponding patching stage first."
    exit 1
fi

echo "Constructing graphs for experiment: $EXPERIMENT"
echo "Variant: $GRAPH_VARIANT"
echo "Input: $PATCH_MATRIX_DIR"
echo "Output: $OUTPUT_DIR"

LAYER_AGG="${LAYER_AGG:-max_abs}"
SELECTION_METHOD="${SELECTION_METHOD:-fdr}"
TOP_K="${TOP_K:-5}"
QUANTILE="${QUANTILE:-0.9}"

if [ -z "${FDR_Q+x}" ]; then
    if [ "$IS_NOPAIR_VARIANT" -eq 1 ]; then
        FDR_Q="0.2"
    else
        FDR_Q="0.2"
    fi
fi

if [ -z "${RELATIVE_EDGE_THRESHOLD+x}" ]; then
    if [ "$IS_NOPAIR_VARIANT" -eq 1 ]; then
        RELATIVE_EDGE_THRESHOLD="0.45"
    else
        RELATIVE_EDGE_THRESHOLD="0.3"
    fi
fi

if [ -z "${PARENT_CAUSAL_RULE+x}" ]; then
    if [ "$IS_NOPAIR_VARIANT" -eq 1 ]; then
        PARENT_CAUSAL_RULE="token_filter_then_relative"
    else
        PARENT_CAUSAL_RULE="strongest_plus_relative"
    fi
fi

EDGE_BUILD_SCOPE="${EDGE_BUILD_SCOPE:-all_nodes}"
STRONGEST_MIN_WEIGHT="${STRONGEST_MIN_WEIGHT:-0.0}"
GRAPH_RENDER="${GRAPH_RENDER:-png}"
GRAPH_LAYOUT="${GRAPH_LAYOUT:-dot}"

echo "Graph config: layer_agg=$LAYER_AGG selection=$SELECTION_METHOD fdr_q=$FDR_Q rel_edge=$RELATIVE_EDGE_THRESHOLD parent_rule=$PARENT_CAUSAL_RULE edge_scope=$EDGE_BUILD_SCOPE strongest_min=$STRONGEST_MIN_WEIGHT render=$GRAPH_RENDER"

python -u construct_graph.py \
  --patch-runs-dir "$PATCH_MATRIX_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --render "$GRAPH_RENDER" \
  --layout "$GRAPH_LAYOUT" \
  --layer-agg "$LAYER_AGG" \
  --selection-method "$SELECTION_METHOD" \
  --top-k "$TOP_K" \
  --quantile "$QUANTILE" \
  --fdr-q "$FDR_Q" \
  --relative-edge-threshold "$RELATIVE_EDGE_THRESHOLD" \
  --parent-causal-rule "$PARENT_CAUSAL_RULE" \
  --edge-build-scope "$EDGE_BUILD_SCOPE" \
  --strongest-min-weight "$STRONGEST_MIN_WEIGHT"
