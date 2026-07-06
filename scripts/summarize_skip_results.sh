#!/bin/bash
# Summarize node_skipping_results.json across experiments for one model.

set -euo pipefail

MODEL_NAME="${1:-Qwen2.5-32B}"
if [[ -z "${WRODERI_SCRATCH_ROOT:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
    source "$SCRIPT_DIR/../workspace_paths.sh"
fi
SCRATCH_ROOT="${2:-$WRODERI_SCRATCH_ROOT}"
BASE_DIR="$SCRATCH_ROOT/traces/$MODEL_NAME"

if [[ ! -d "$BASE_DIR" ]]; then
    echo "ERROR: Model traces directory not found: $BASE_DIR"
    exit 1
fi

python - "$BASE_DIR" <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
rows = []
missing = []

for exp_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
    path = exp_dir / "node_skipping_results.json"
    if not path.exists():
        missing.append(exp_dir.name)
        continue
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        rows.append((exp_dir.name, "ERR", str(exc)))
        continue

    metrics = payload.get("metrics", {})
    rows.append((
        exp_dir.name,
        int(metrics.get("total_tests", 0)),
        int(metrics.get("successful_skips", 0)),
        int(metrics.get("correct_after_skip", 0)),
        float(metrics.get("success_rate", 0.0)),
        float(metrics.get("robustness_score", 0.0)),
    ))

print("\nNODE SKIPPING SUMMARY")
print("=" * 96)
print(f"{'experiment':34s} {'tests':>6s} {'success':>8s} {'correct':>8s} {'success%':>10s} {'robust%':>10s}")
print("-" * 96)

ok_rows = [r for r in rows if len(r) == 6]
for r in rows:
    if len(r) != 6:
        exp, _, err = r
        print(f"{exp:34s} {'ERR':>6s} {'-':>8s} {'-':>8s} {'-':>10s} {'-':>10s}")
        continue
    exp, tests, success, correct, success_rate, robust = r
    print(f"{exp:34s} {tests:6d} {success:8d} {correct:8d} {100*success_rate:10.2f} {100*robust:10.2f}")

print("-" * 96)
if ok_rows:
    total_tests = sum(r[1] for r in ok_rows)
    total_success = sum(r[2] for r in ok_rows)
    total_correct = sum(r[3] for r in ok_rows)
    success_rate = (total_success / total_tests) if total_tests else 0.0
    robust = (total_correct / total_success) if total_success else 0.0
    print(f"{'TOTAL':34s} {total_tests:6d} {total_success:8d} {total_correct:8d} {100*success_rate:10.2f} {100*robust:10.2f}")
else:
    print("No node_skipping_results.json files found.")

if missing:
    print("\nMissing results:")
    print(", ".join(missing))
print()
PY
