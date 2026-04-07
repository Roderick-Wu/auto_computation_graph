#!/bin/bash
# Run both result summary scripts.

set -euo pipefail

MODEL_NAME="${1:-Qwen2.5-32B}"
SCRATCH_ROOT="${2:-$HOME/scratch}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/summarize_causal_results.sh" "$MODEL_NAME" "$SCRATCH_ROOT"
bash "$SCRIPT_DIR/summarize_skip_results.sh" "$MODEL_NAME" "$SCRATCH_ROOT"
