#!/bin/bash
set -euo pipefail

THIS_SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
THIS_DIR="$(cd "$(dirname "$THIS_SCRIPT_PATH")" && pwd)"
source "$THIS_DIR/../workspace_paths.sh"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-32B}"
RENDER="${RENDER:-none}"
LAYOUT="${LAYOUT:-dot}"
REQUEST_SLEEP_SECONDS="${REQUEST_SLEEP_SECONDS:-0.0}"
EXCLUDE_SHARED_VALUES="${EXCLUDE_SHARED_VALUES:-1}"
USE_JSON_MODE="${USE_JSON_MODE:-1}"
OVERWRITE="${OVERWRITE:-0}"

cd "$THIS_DIR"

while IFS=$'\t' read -r experiment _n_formats; do
  [ -z "$experiment" ] && continue

  aligned_pairs_json="$WRODERI_SCRATCH_ROOT/traces/$MODEL_NAME/$experiment/aligned_pairs.json"
  output_dir="$WRODERI_SCRATCH_ROOT/traces/$MODEL_NAME/$experiment/graphs_ground_truth_api"

  if [ ! -f "$aligned_pairs_json" ]; then
    echo "Skipping $experiment: missing $aligned_pairs_json"
    continue
  fi

  args=(
    --model-name "$MODEL_NAME"
    --experiment "$experiment"
    --aligned-pairs-json "$aligned_pairs_json"
    --output-dir "$output_dir"
    --render "$RENDER"
    --layout "$LAYOUT"
    --request-sleep-seconds "$REQUEST_SLEEP_SECONDS"
  )

  if [ "$EXCLUDE_SHARED_VALUES" = "1" ]; then
    args+=(--exclude-shared-values)
  fi

  if [ "$USE_JSON_MODE" = "1" ]; then
    args+=(--use-json-mode)
  fi

  if [ "$OVERWRITE" = "1" ]; then
    args+=(--overwrite)
  fi

  echo
  echo "=== Ground-truth API generation: model=$MODEL_NAME experiment=$experiment ==="
  python generate_ground_truth_graphs_api.py "${args[@]}"
done < <(python "$THIS_DIR/list_all_experiments.py")

echo
echo "Completed ground-truth API sweep for $MODEL_NAME."
