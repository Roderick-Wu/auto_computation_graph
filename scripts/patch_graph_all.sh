#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../workspace_paths.sh"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-72B}"
PATCH_BATCH_SIZE="${PATCH_BATCH_SIZE:-16}"
TOKEN_POSITIONS_TO_PATCH="${TOKEN_POSITIONS_TO_PATCH:-all}"
H100_GPUS="${H100_GPUS:-2}"

submitted=0
skipped=0

while read -r experiment n_formats
do
    [ -z "$experiment" ] && continue

    input_json="$WRODERI_SCRATCH_ROOT/traces/$MODEL_NAME/$experiment/aligned_pairs.json"
    if [ ! -f "$input_json" ]; then
        echo "Skipping ${experiment}: missing $input_json"
        skipped=$((skipped + 1))
        continue
    fi

    pair_count="$(
        python - "$input_json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    payload = json.load(f)

if isinstance(payload, list):
    print(len(payload))
elif isinstance(payload, dict) and isinstance(payload.get("pairs"), list):
    print(len(payload["pairs"]))
else:
    print(0)
PY
    )"

    if [ "$pair_count" -le 0 ]; then
        echo "Skipping ${experiment}: $pair_count aligned pairs"
        skipped=$((skipped + 1))
        continue
    fi

    echo "Submitting counterfactual token-swap job for ${experiment} (${n_formats} formats, ${pair_count} aligned pairs) on ${MODEL_NAME}"
    sbatch \
        --gpus-per-node="h100:$H100_GPUS" \
        --export=ALL,MODEL_NAME="$MODEL_NAME",PATCH_BATCH_SIZE="$PATCH_BATCH_SIZE",TOKEN_POSITIONS_TO_PATCH="$TOKEN_POSITIONS_TO_PATCH" \
        "$SCRIPT_DIR/patch_graph.sh" "$experiment" "$MODEL_NAME"
    submitted=$((submitted + 1))
done < <(python "$WRODERI_REPO_ROOT/src/list_all_experiments.py")

echo "Counterfactual token-swap submission complete for ${MODEL_NAME}: submitted=${submitted} skipped=${skipped}"
