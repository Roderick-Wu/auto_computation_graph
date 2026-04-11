#!/bin/bash
#SBATCH --job-name=patch_graph_nopair
#SBATCH --time=0-48:00:00 # D-HH:MM
#SBATCH --account=def-rgrosse
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1

#salloc --account=def-zhijing --mem=512G --gpus=h100:2

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch patch_graph_nopair.sh <experiment> [model_name]"
    exit 1
fi

experiment="$1"
model_name="${2:-Qwen2.5-32B}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load python/3.11.5 cuda/12.6 scipy-stack/2023b arrow/21.0.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

echo "Running no-pair patching on experiment: $experiment"
echo "Input: $input_json"
echo "Output: $output_root_dir"

python -u intervene_graph_nopair.py \
    --input-json "$input_json" \
    --output-root-dir "$output_root_dir" \
    --model-path "$model_path" \
    --tokenizer-path "$model_path" \
    --device cuda \
    --dtype float16 \
    --patch-batch-size 16
