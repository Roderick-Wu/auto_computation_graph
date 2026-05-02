#!/bin/bash
set -euo pipefail

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
if [ -z "$SCRIPT_DIR" ] || [ "$SCRIPT_DIR" = "$BASH_SOURCE" ]; then
  SCRIPT_DIR="."
fi
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"

source "$SCRIPT_DIR/../workspace_paths.sh"

module load python/3.11.5
module load scipy-stack/2023b
module load arrow/21.0.0

MODEL_NAME="${1:-Qwen2.5-32B}"
EXPERIMENT="${2:-velocity_from_ke}"
MAX_PAIRS="${3:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-$WRODERI_SCRATCH_ROOT/traces/$MODEL_NAME/$EXPERIMENT/graphs_ground_truth_api}"
ALIGNED_PAIRS_JSON="${ALIGNED_PAIRS_JSON:-$WRODERI_SCRATCH_ROOT/traces/$MODEL_NAME/$EXPERIMENT/aligned_pairs.json}"

EXCLUDE_SHARED_VALUES="${EXCLUDE_SHARED_VALUES:-1}"
USE_JSON_MODE="${USE_JSON_MODE:-1}"
RENDER="${RENDER:-none}"
LAYOUT="${LAYOUT:-dot}"
REQUEST_SLEEP_SECONDS="${REQUEST_SLEEP_SECONDS:-0.0}"

cd "$SCRIPT_DIR"

ARGS=(
  --model-name "$MODEL_NAME"
  --experiment "$EXPERIMENT"
  --aligned-pairs-json "$ALIGNED_PAIRS_JSON"
  --output-dir "$OUTPUT_DIR"
  --render "$RENDER"
  --layout "$LAYOUT"
  --request-sleep-seconds "$REQUEST_SLEEP_SECONDS"
)

if [ "$MAX_PAIRS" -gt 0 ]; then
  ARGS+=(--max-pairs "$MAX_PAIRS")
fi

if [ "$EXCLUDE_SHARED_VALUES" = "1" ]; then
  ARGS+=(--exclude-shared-values)
fi

if [ "$USE_JSON_MODE" = "1" ]; then
  ARGS+=(--use-json-mode)
fi

echo "Generating API ground-truth graphs:"
echo "  model      : $MODEL_NAME"
echo "  experiment : $EXPERIMENT"
echo "  input      : $ALIGNED_PAIRS_JSON"
echo "  output     : $OUTPUT_DIR"
echo "  max_pairs  : $MAX_PAIRS (0 means all)"
echo "  render     : $RENDER"

python generate_ground_truth_graphs_api.py "${ARGS[@]}"
