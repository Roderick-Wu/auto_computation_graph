#!/bin/bash
#SBATCH --job-name=download_model
#SBATCH --time=2-00:00:00
#SBATCH --account=def-rgrosse
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: sbatch download_model_job.sh <hf_repo_id> <local_model_name>"
    exit 1
fi

HF_REPO_ID="$1"
LOCAL_MODEL_NAME="$2"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load python/3.11.5
module load scipy-stack/2023b
module load arrow/21.0.0

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "$SCRIPT_DIR/../workspace_paths.sh"

TARGET_DIR="$WRODERI_MODELS_ROOT/$LOCAL_MODEL_NAME"
mkdir -p "$TARGET_DIR"

echo "Downloading HuggingFace model:"
echo "  repo:   $HF_REPO_ID"
echo "  target: $TARGET_DIR"

python - "$HF_REPO_ID" "$TARGET_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
target_dir = sys.argv[2]

snapshot_download(
    repo_id=repo_id,
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)

print(f"Completed download: {repo_id} -> {target_dir}")
PY
