#!/bin/bash
# Run both result summary scripts.

set -euo pipefail

MODEL_NAME="${1:-Qwen2.5-32B}"
if [[ -z "${WRODERI_SCRATCH_ROOT:-}" ]]; then
	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	source "$SCRIPT_DIR/../workspace_paths.sh"
fi
SCRATCH_ROOT="${2:-$WRODERI_SCRATCH_ROOT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/summarize_causal_results.sh" "$MODEL_NAME" "$SCRATCH_ROOT"
bash "$SCRIPT_DIR/summarize_skip_results.sh" "$MODEL_NAME" "$SCRATCH_ROOT"
