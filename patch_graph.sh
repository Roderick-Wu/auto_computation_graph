#!/bin/bash
#SBATCH --job-name=patch_graph
#SBATCH --time=0-8:00:00 # D-HH:MM
#SBATCH --account=def-rgrosse
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=1

#salloc --account=def-zhijing --mem=512G --gpus=h100:2

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Load required modules
#module load python cuda scipy-stack arrow
module load python/3.11.5 cuda/12.6 scipy-stack/2023b arrow/21.0.0

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/../workspace_paths.sh"

#source venv/bin/activate

# Default experiment (e.g., "velocity")
EXPERIMENT=${1:-velocity}
MODEL_NAME=${2:-${MODEL_NAME:-Qwen2.5-72B}}
LAYER_STRIDE=${LAYER_STRIDE:-1}

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/../workspace_paths.sh"

# Input/output paths
TRACES_DIR="$WRODERI_SCRATCH_ROOT/traces/${MODEL_NAME}/${EXPERIMENT}"
INPUT_JSON="${TRACES_DIR}/aligned_pairs.json"
OUTPUT_ROOT_DIR="${TRACES_DIR}/patch_runs"
#OUTPUT_ROOT_DIR="${TRACES_DIR}/patch_runs_nopair"

# Check if input exists
if [ ! -f "$INPUT_JSON" ]; then
    echo "ERROR: Input file not found: $INPUT_JSON"
    echo "Please run post_process_pairs.py first to generate aligned_pairs.json"
    exit 1
fi

echo "Running patching on experiment: $EXPERIMENT"
echo "Input: $INPUT_JSON"
echo "Output: $OUTPUT_ROOT_DIR"
echo "Layer stride: $LAYER_STRIDE"

python -u intervene_graph.py \
  --input-json "$INPUT_JSON" \
  --output-root-dir "$OUTPUT_ROOT_DIR" \
  --device cuda \
  --dtype float16 \
  --layer-stride "$LAYER_STRIDE" \
  --patch-batch-size 16

#python -u intervene_graph_nopair.py \
  #--input-json "$INPUT_JSON" \
  #--output-root-dir "$OUTPUT_ROOT_DIR" \
  #--device cuda \
  #--dtype float16 \
  #--patch-batch-size 16