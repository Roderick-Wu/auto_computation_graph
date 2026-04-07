#!/bin/bash
#SBATCH --job-name=validate_causal_structure
#SBATCH --time=0-04:00:00 # D-HH:MM
#SBATCH --account=def-rgrosse
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100:1
# Validate causal structure on a single pair via token-level intervention

set -euo pipefail

module load python/3.11.5
module load cuda/12.6
module load scipy-stack/2023b
module load arrow/21.0.0

MODEL_NAME="${1:-Qwen2.5-32B}"
EXPERIMENT="${2:-velocity}"
GRAPH_DIR="${3:-}"
MAX_PAIRS="${4:-5}"

cd "/home/wuroderi/projects/def-zhijing/wuroderi/auto_computation_graph"

if [[ -z "$GRAPH_DIR" ]]; then
    SCRATCH_ROOT="$HOME/scratch"
    GRAPH_DIR="$SCRATCH_ROOT/traces/$MODEL_NAME/$EXPERIMENT/graphs"
fi

echo "Testing causal structure for: $EXPERIMENT on $MODEL_NAME"
echo "Graph directory: $GRAPH_DIR"

python intervene_validate_causal_structure.py \
    --model-name "$MODEL_NAME" \
    --experiment "$EXPERIMENT" \
    --graph-dir "$GRAPH_DIR" \
    --max-pairs "$MAX_PAIRS" \
    --device cuda \
    --dtype float16

echo "Validation complete. Check causal_validation_results.json in experiment logs."
