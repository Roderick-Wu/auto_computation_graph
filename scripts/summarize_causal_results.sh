#!/bin/bash
# Summarize causal_validation_results.json across experiments for one model.

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
    path = exp_dir / "causal_validation_results.json"
    if not path.exists():
        missing.append(exp_dir.name)
        continue
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        rows.append((exp_dir.name, "ERR", str(exc)))
        continue

    metrics = payload.get("metrics", {})
    results = payload.get("results", []) if isinstance(payload.get("results"), list) else []
    null_pairs = 0
    for r in results:
        if r.get("baseline_answer") is None and r.get("intervened_answer") is None:
            null_pairs += 1

    rows.append((
        exp_dir.name,
        int(metrics.get("total_tests", 0)),
        int(metrics.get("positive_control_tests", 0)),
        int(metrics.get("negative_control_tests", 0)),
        float(metrics.get("positive_hit_rate", 0.0)),
        float(metrics.get("negative_false_alarm_rate", 0.0)),
        float(metrics.get("specificity_score", 0.0)),
        float(metrics.get("sensitivity_score", 0.0)),
        null_pairs,
    ))

print("\nCAUSAL VALIDATION SUMMARY")
print("=" * 110)
print(f"{'experiment':34s} {'tests':>6s} {'pos':>5s} {'neg':>5s} {'hit%':>8s} {'fa%':>8s} {'spec%':>8s} {'sens%':>8s} {'null/null':>10s}")
print("-" * 110)

ok_rows = [r for r in rows if len(r) == 9]
for r in rows:
    if len(r) != 9:
        exp, _, err = r
        print(f"{exp:34s} {'ERR':>6s} {'-':>5s} {'-':>5s} {'-':>8s} {'-':>8s} {'-':>8s} {'-':>8s} {'-':>10s}")
        continue
    exp, tests, pos, neg, hit, fa, spec, sens, null_pairs = r
    print(f"{exp:34s} {tests:6d} {pos:5d} {neg:5d} {100*hit:8.2f} {100*fa:8.2f} {100*spec:8.2f} {100*sens:8.2f} {null_pairs:10d}")

print("-" * 110)
if ok_rows:
    total_tests = sum(r[1] for r in ok_rows)
    total_pos = sum(r[2] for r in ok_rows)
    total_neg = sum(r[3] for r in ok_rows)
    weighted_hit = (sum(r[4] * r[2] for r in ok_rows) / total_pos) if total_pos else 0.0
    weighted_fa = (sum(r[5] * r[3] for r in ok_rows) / total_neg) if total_neg else 0.0
    weighted_spec = 1.0 - weighted_fa
    weighted_sens = weighted_hit
    total_null_pairs = sum(r[8] for r in ok_rows)
    print(f"{'TOTAL':34s} {total_tests:6d} {total_pos:5d} {total_neg:5d} {100*weighted_hit:8.2f} {100*weighted_fa:8.2f} {100*weighted_spec:8.2f} {100*weighted_sens:8.2f} {total_null_pairs:10d}")
else:
    print("No causal_validation_results.json files found.")

if missing:
    print("\nMissing results:")
    print(", ".join(missing))
print()
PY
