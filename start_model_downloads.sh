#!/bin/bash
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SELF_DIR/../workspace_paths.sh"

cd "$SELF_DIR"

module load python/3.11.5
module load scipy-stack/2023b
module load arrow/21.0.0

mkdir -p smoke_logs

download_one() {
    local repo_id="$1"
    local model_name="$2"
    local target_dir="$WRODERI_MODELS_ROOT/$model_name"
    local log_file="$SELF_DIR/smoke_logs/download_${model_name}.log"

    mkdir -p "$target_dir"

    echo "[$(date -Is)] START ${model_name} (${repo_id})" | tee -a "$log_file"
    python - "$repo_id" "$target_dir" >>"$log_file" 2>&1 <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
target_dir = sys.argv[2]

print(f"Downloading {repo_id} -> {target_dir}")
snapshot_download(
    repo_id=repo_id,
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f"Done: {repo_id}")
PY
    echo "[$(date -Is)] DONE ${model_name}" | tee -a "$log_file"
}

download_one "google/gemma-4-31b-it" "gemma-4-31B-it"
download_one "meta-llama/Llama-3.1-70B" "Llama-3.1-70B"
download_one "openai/gpt-oss-20b" "gpt-oss-20b"
download_one "mistralai/Mistral-Small-3.1-24B-Base-2503" "Mistral-Small-3.1-24B-Base-2503"

echo "All requested model downloads completed."
