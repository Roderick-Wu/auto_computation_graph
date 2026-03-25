#!/bin/bash
#SBATCH --job-name=reasoning_abstraction
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

source venv/bin/activate

python intervene_graph.py

