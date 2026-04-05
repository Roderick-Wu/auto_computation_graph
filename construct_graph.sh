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

# Default experiment (e.g., "velocity")
EXPERIMENT=${1:-velocity}
MODEL_NAME=${2:-Qwen2.5-32B}

# Input/output paths
TRACES_DIR="/home/wuroderi/scratch/traces/${MODEL_NAME}/${EXPERIMENT}"
PATCH_RUNS_DIR="${TRACES_DIR}/patch_runs"
OUTPUT_DIR="${TRACES_DIR}/graphs"

# Check if patch_runs directory exists
if [ ! -d "$PATCH_RUNS_DIR" ]; then
    echo "ERROR: Patch runs directory not found: $PATCH_RUNS_DIR"
    echo "Please run intervene_graph.py first to generate patch_runs"
    exit 1
fi

echo "Constructing graphs for experiment: $EXPERIMENT"
echo "Input: $PATCH_RUNS_DIR"
echo "Output: $OUTPUT_DIR"

python -u construct_graph.py \
  --patch-runs-dir "$PATCH_RUNS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --render png \
  --layout dot \
  --layer-agg max_abs \
  --selection-method fdr \
  --fdr-q 0.1 \
  --relative-edge-threshold 0.65