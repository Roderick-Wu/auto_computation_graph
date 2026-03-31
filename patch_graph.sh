#!/bin/bash
#SBATCH --job-name=patch_graph
#SBATCH --time=0-8:00:00 # D-HH:MM
#SBATCH --account=def-rgrosse
#SBATCH --mem=256G
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1

#salloc --account=def-zhijing --mem=512G --gpus=h100:2

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Load required modules
#module load python cuda scipy-stack arrow
module load python/3.11.5 cuda/12.6 scipy-stack/2023b arrow/21.0.0

#source venv/bin/activate

# Default experiment (e.g., "velocity")
EXPERIMENT=${1:-velocity}
MODEL_NAME=${2:-Qwen2.5-32B}

# Input/output paths
TRACES_DIR="/home/wuroderi/links/scratch/traces/${MODEL_NAME}/${EXPERIMENT}"
INPUT_JSON="${TRACES_DIR}/aligned_pairs.json"
OUTPUT_ROOT_DIR="${TRACES_DIR}/patch_runs"

# Check if input exists
if [ ! -f "$INPUT_JSON" ]; then
    echo "ERROR: Input file not found: $INPUT_JSON"
    echo "Please run post_process_pairs.py first to generate aligned_pairs.json"
    exit 1
fi

echo "Running patching on experiment: $EXPERIMENT"
echo "Input: $INPUT_JSON"
echo "Output: $OUTPUT_ROOT_DIR"

python intervene_graph.py \
  --input-json "$INPUT_JSON" \
  --output-root-dir "$OUTPUT_ROOT_DIR" \
  --device cuda \
  --dtype float16

