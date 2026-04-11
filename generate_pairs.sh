#!/bin/bash
#SBATCH --job-name=generate_pairs
#SBATCH --time=0-4:00:00
#SBATCH --account=def-rgrosse
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch generate_pairs.sh <experiment> [model_name]"
    exit 1
fi

experiment="$1"
model_name="${2:-Qwen2.5-32B}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load python/3.11.5
module load scipy-stack/2023b
module load arrow/21.0.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../workspace_paths.sh"

traces_json="$WRODERI_SCRATCH_ROOT/traces/$model_name/$experiment/reject_traces.json"
if [ ! -f "$traces_json" ]; then
    legacy_traces_json="$WRODERI_SCRATCH_ROOT/traces/$model_name/$experiment/fixed_traces.json"
    if [ -f "$legacy_traces_json" ]; then
        traces_json="$legacy_traces_json"
    fi
fi

if [ -d venv ]; then
    source venv/bin/activate
fi

cd "$WRODERI_PROJECT_ROOT/auto_computation_graph"

python intervene_generate_pairs.py \
    --model-name "$model_name" \
    --experiment "$experiment" \
    --traces-json "$traces_json" \
