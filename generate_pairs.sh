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

scratch_root="$HOME/links/scratch"
if [ ! -d "$scratch_root" ]; then
    scratch_root="$HOME/scratch"
fi

traces_json="$scratch_root/traces/$model_name/$experiment/fixed_traces.json"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load python/3.11.5
module load scipy-stack/2023b
module load arrow/21.0.0

if [ -d venv ]; then
    source venv/bin/activate
fi

python intervene_generate_pairs.py \
    --model-name "$model_name" \
    --experiment "$experiment" \
    --traces-json "$traces_json"
