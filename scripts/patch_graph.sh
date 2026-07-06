#!/bin/bash
#SBATCH --job-name=patch_graph
#SBATCH --time=0-8:00:00 # D-HH:MM
#SBATCH --account=def-rgrosse
#SBATCH --mem=256G
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python/3.11.5 cuda/12.6 scipy-stack/2023b arrow/21.0.0

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/../workspace_paths.sh"

EXPERIMENT=${1:-velocity}
MODEL_NAME=${2:-${MODEL_NAME:-Qwen2.5-72B}}
PATCH_BATCH_SIZE=${PATCH_BATCH_SIZE:-16}
TOKEN_POSITIONS_TO_PATCH=${TOKEN_POSITIONS_TO_PATCH:-all}

TRACES_DIR="$WRODERI_SCRATCH_ROOT/traces/${MODEL_NAME}/${EXPERIMENT}"
INPUT_JSON="${TRACES_DIR}/aligned_pairs.json"
OUTPUT_ROOT_DIR="${TRACES_DIR}/patch_runs"
MODEL_PATH="$WRODERI_MODELS_ROOT/$MODEL_NAME"

if [ ! -f "$INPUT_JSON" ]; then
    echo "ERROR: Input file not found: $INPUT_JSON"
    echo "Please run post_process_pairs.py first to generate aligned_pairs.json"
    exit 1
fi

echo "Running counterfactual token-swap patching on experiment: $EXPERIMENT"
echo "Input: $INPUT_JSON"
echo "Output: $OUTPUT_ROOT_DIR"
echo "Patch batch size: $PATCH_BATCH_SIZE"
echo "Token positions: $TOKEN_POSITIONS_TO_PATCH"

python -u "$WRODERI_REPO_ROOT/src/intervene_graph.py" \
  --input-json "$INPUT_JSON" \
  --output-root-dir "$OUTPUT_ROOT_DIR" \
  --model-path "$MODEL_PATH" \
  --tokenizer-path "$MODEL_PATH" \
  --device cuda \
  --dtype float16 \
  --token-positions-to-patch "$TOKEN_POSITIONS_TO_PATCH" \
  --patch-batch-size "$PATCH_BATCH_SIZE"
