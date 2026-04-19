#!/bin/bash
#SBATCH --job-name=patch_graph_nopair
#SBATCH --time=0-8:00:00 # D-HH:MM
#SBATCH --account=def-rgrosse
#SBATCH --mem=256G
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=1

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch patch_graph_nopair.sh <experiment> [model_name]"
    exit 1
fi

experiment="$1"
model_name="${2:-${MODEL_NAME:-Qwen2.5-32B}}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-16}"
TOKEN_POSITIONS_TO_PATCH="${TOKEN_POSITIONS_TO_PATCH:-all}"
NOISE_MODE="${NOISE_MODE:-gaussian}"
NOISE_SCALE="${NOISE_SCALE:-1.0}"
NOISE_SEED="${NOISE_SEED:-0}"
NOISE_SAMPLES_PER_TOKEN="${NOISE_SAMPLES_PER_TOKEN:-10}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python/3.11.5 cuda/12.6 scipy-stack/2023b arrow/21.0.0

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/../workspace_paths.sh"

traces_dir="$WRODERI_SCRATCH_ROOT/traces/$model_name/$experiment"
input_json="$traces_dir/reject_traces.json"
if [ ! -f "$input_json" ]; then
    legacy_traces_json="$traces_dir/fixed_traces.json"
    if [ -f "$legacy_traces_json" ]; then
        input_json="$legacy_traces_json"
    fi
fi

if [ ! -f "$input_json" ]; then
    echo "ERROR: Input file not found: $input_json"
    echo "Please run reject_traces.py first to generate reject_traces.json"
    exit 1
fi

output_root_dir="$traces_dir/patch_solo"
model_path="$WRODERI_MODELS_ROOT/$model_name"

echo "Running no-pair token-noise patching on experiment: $experiment"
echo "Input: $input_json"
echo "Output: $output_root_dir"
echo "Token positions: $TOKEN_POSITIONS_TO_PATCH"
echo "Patch batch size: $PATCH_BATCH_SIZE"
echo "Noise mode: $NOISE_MODE"
echo "Noise scale: $NOISE_SCALE"
echo "Noise seed: $NOISE_SEED"
echo "Noise samples/token: $NOISE_SAMPLES_PER_TOKEN"

python -u intervene_graph_nopair.py \
    --input-json "$input_json" \
    --output-root-dir "$output_root_dir" \
    --model-path "$model_path" \
    --tokenizer-path "$model_path" \
    --device cuda \
    --dtype float16 \
    --token-positions-to-patch "$TOKEN_POSITIONS_TO_PATCH" \
    --noise-mode "$NOISE_MODE" \
    --noise-scale "$NOISE_SCALE" \
    --noise-seed "$NOISE_SEED" \
    --noise-samples-per-token "$NOISE_SAMPLES_PER_TOKEN" \
    --patch-batch-size "$PATCH_BATCH_SIZE"
