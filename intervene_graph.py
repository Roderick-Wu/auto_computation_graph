"""Counterfactual token-swap patching runner.

For each experiment entry (either precomputed experiments or built from aligned pairs):
1. Run BASE (original/source) forward pass and score target logprobs.
2. For every token position before truncation, swap in the aligned counterfactual token.
3. Re-run and measure delta logprob at the truncation target.
4. Save one token-level matrix row (shape [1, positions]) per value.

This refactor intentionally removes layer-wise intervention and performs direct
input-token interventions only.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workspace_paths import resolve_auto_traces_root, resolve_model_path

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

INPUT_JSON = resolve_auto_traces_root("Qwen2.5-32B", "velocity") / "aligned_pairs.json"
OUTPUT_ROOT_DIR = resolve_auto_traces_root("Qwen2.5-32B", "velocity") / "patch_runs"
OUTPUT_SUMMARY_JSON = OUTPUT_ROOT_DIR / "patching_summary.json"

EXPERIMENT_INDICES = None
EXPERIMENT_IDS = None
TOKEN_POSITIONS_TO_PATCH = None
PATCH_BATCH_SIZE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

MODEL_PATH = str(resolve_model_path("Qwen2.5-32B"))
TOKENIZER_PATH = MODEL_PATH
SAVE_PLOTS = True
RESUME = True


def parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    text = raw.strip()
    if text == "" or text.lower() == "all":
        return None
    out: List[int] = []
    for chunk in text.split(","):
        part = chunk.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def parse_str_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    text = raw.strip()
    if text == "" or text.lower() == "all":
        return None
    out: List[str] = []
    for chunk in text.split(","):
        part = chunk.strip()
        if part:
            out.append(part)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run token-level counterfactual swapping and export matrices.")
    parser.add_argument("--input-json", type=Path, default=INPUT_JSON, help="Input JSON with top-level 'pairs' or 'experiments'.")
    parser.add_argument("--output-root-dir", type=Path, default=OUTPUT_ROOT_DIR, help="Output root directory for run artifacts.")
    parser.add_argument("--output-summary-json", type=Path, default=None, help="Optional summary JSON path (default: <output-root-dir>/patching_summary.json).")

    parser.add_argument("--experiment-indices", type=str, default=None, help="Comma-separated experiment indices, e.g. '0,1,2'.")
    parser.add_argument("--experiment-ids", type=str, default=None, help="Comma-separated experiment IDs.")
    parser.add_argument("--token-positions-to-patch", type=str, default=None, help="Comma-separated token positions, or 'all'.")
    parser.add_argument("--patch-batch-size", type=int, default=PATCH_BATCH_SIZE, help="Number of token positions to intervene per forward pass.")

    # Legacy/compatibility knobs accepted but ignored in token-only mode.
    parser.add_argument("--layers-to-patch", type=str, default=None, help="Legacy arg; ignored in token-only mode.")
    parser.add_argument("--layer-stride", type=int, default=1, help="Legacy arg; ignored in token-only mode.")
    parser.add_argument("--patch-scope", type=str, default="token", help="Legacy arg; ignored in token-only mode.")

    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Model path for AutoModelForCausalLM.")
    parser.add_argument("--tokenizer-path", type=str, default=TOKENIZER_PATH, help="Tokenizer path for AutoTokenizer.")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to run on (e.g., cuda, cpu).")
    parser.add_argument("--dtype", type=str, default="float16" if DTYPE == torch.float16 else "float32", choices=["float16", "float32"], help="Model dtype.")
    parser.add_argument("--no-plots", action="store_true", help="Disable heatmap PNG generation and only write JSON outputs.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume behavior and recompute all selected experiments.")
    return parser


def load_existing_pair_payloads(output_root_dir: Path) -> Dict[str, Dict]:
    existing: Dict[str, Dict] = {}
    if not output_root_dir.exists():
        return existing

    for pair_dir in sorted(output_root_dir.glob("pair*")):
        if not pair_dir.is_dir():
            continue
        pair_json = pair_dir / "pair_matrices.json"
        if not pair_json.exists():
            continue
        try:
            with open(pair_json, "r") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
                existing[pair_dir.name] = payload
        except Exception as e:
            print(f"  Warning: failed to load existing pair JSON {pair_json}: {e}")
    return existing


def collect_completed_experiment_ids(pair_payloads: Dict[str, Dict]) -> set:
    completed = set()
    for payload in pair_payloads.values():
        for entry in payload.get("entries", []):
            exp_id = entry.get("experiment_id")
            if isinstance(exp_id, str):
                completed.add(exp_id)
    return completed


def merge_run_records(existing_runs: List[Dict], new_runs: List[Dict]) -> List[Dict]:
    merged_by_exp: Dict[str, Dict] = {}
    for rec in existing_runs:
        exp_id = rec.get("experiment_id")
        if isinstance(exp_id, str):
            merged_by_exp[exp_id] = rec
    for rec in new_runs:
        exp_id = rec.get("experiment_id")
        if isinstance(exp_id, str):
            merged_by_exp[exp_id] = rec

    merged = list(merged_by_exp.values())
    merged.sort(key=lambda r: (str(r.get("pair_id", "")), int(r.get("truncation_token_index", -1))), reverse=True)
    return merged


def build_experiments_from_pairs(payload: Dict) -> List[Dict]:
    def _values_differ(source_meta: Dict, cf_meta: Dict) -> bool:
        src_norm = source_meta.get("normalized_value")
        cf_norm = cf_meta.get("normalized_value")
        if src_norm is not None and cf_norm is not None:
            return str(src_norm) != str(cf_norm)

        src_text = source_meta.get("value_text")
        cf_text = cf_meta.get("value_text")
        if src_text is not None and cf_text is not None:
            return str(src_text) != str(cf_text)

        return False

    pairs = payload.get("pairs")
    if not isinstance(pairs, list):
        return []

    experiments: List[Dict] = []
    for pidx, pair in enumerate(pairs):
        pair_id = pair.get("id", pidx)
        source_block = pair.get("pair", {}).get("source", {})
        cf_block = pair.get("pair", {}).get("counterfactual", {})
        source_values = source_block.get("values") if isinstance(source_block.get("values"), list) else []

        token_counts_equal = pair.get("post_process", {}).get("token_counts_equal")
        if token_counts_equal is False:
            continue

        # BASE := source/original trace, CF := counterfactual trace.
        base_token_ids = source_block.get("tokens") if isinstance(source_block.get("tokens"), list) else []
        cf_token_ids = cf_block.get("tokens") if isinstance(cf_block.get("tokens"), list) else []
        source_text = source_block.get("generated_text", "")
        if not base_token_ids or not cf_token_ids:
            continue
        if len(base_token_ids) != len(cf_token_ids):
            continue

        cot_marker = "Answer (step-by-step)"
        cot_start_pos = source_text.find(cot_marker)
        if cot_start_pos == -1:
            continue
        cot_start_pos += len(cot_marker)

        matched_values = (
            pair.get("post_process", {})
            .get("numeric_length_alignment", {})
            .get("matched_values", [])
        )

        cot_values = []
        if isinstance(matched_values, list):
            for mv in matched_values:
                if not isinstance(mv, dict):
                    continue
                source_span = mv.get("source", {})
                cf_span = mv.get("counterfactual", {})
                span_start = source_span.get("span_start")
                if span_start is None:
                    span_start = source_span.get("span", {}).get("start")
                if span_start is not None and span_start >= cot_start_pos and _values_differ(source_span, cf_span):
                    cot_values.append(mv)

        def _mv_span_start(mv: Dict) -> int:
            src = mv.get("source", {})
            val = src.get("span_start")
            if val is None:
                val = src.get("span", {}).get("start", 0)
            return int(val) if isinstance(val, int) else 0

        cot_values.sort(key=_mv_span_start, reverse=True)

        if not cot_values:
            continue

        def _resolve_source_value_position(mv: Dict) -> Optional[int]:
            pos = mv.get("position")
            if isinstance(pos, int) and 0 <= pos < len(source_values):
                return pos

            src = mv.get("source", {}) if isinstance(mv.get("source"), dict) else {}
            src_ts = src.get("token_start")
            src_te = src.get("token_end")
            src_ss = src.get("span_start")
            src_se = src.get("span_end")
            src_vt = src.get("value_text")

            for idx, sv in enumerate(source_values):
                if not isinstance(sv, dict):
                    continue
                if src_ts is not None and sv.get("token_start") != src_ts:
                    continue
                if src_te is not None and sv.get("token_end") != src_te:
                    continue
                if src_ss is not None and sv.get("span_start") != src_ss:
                    continue
                if src_se is not None and sv.get("span_end") != src_se:
                    continue
                if src_vt is not None and sv.get("value_text") != src_vt:
                    continue
                return idx
            return None

        used_trunc_indices = set()
        for v_idx, mv in enumerate(cot_values):
            src_meta = mv.get("source", {})
            cf_meta = mv.get("counterfactual", {})

            src_tok_start = src_meta.get("token_start")
            src_tok_end = src_meta.get("token_end")
            cf_tok_start = cf_meta.get("token_start")
            cf_tok_end = cf_meta.get("token_end")

            if not isinstance(src_tok_start, int) or not isinstance(cf_tok_start, int):
                continue

            if isinstance(src_tok_end, int) and isinstance(cf_tok_end, int):
                span_start = max(0, min(src_tok_start, cf_tok_start))
                span_end = min(len(base_token_ids), len(cf_token_ids), src_tok_end, cf_tok_end)
            else:
                span_start = max(0, min(src_tok_start, cf_tok_start))
                span_end = span_start + 1

            if span_end <= span_start:
                continue

            trunc_idx = span_start
            found_diff = False
            for t in range(span_start, span_end):
                if base_token_ids[t] != cf_token_ids[t]:
                    trunc_idx = t
                    found_diff = True
                    break

            if not found_diff:
                continue
            if trunc_idx <= 0:
                continue
            if trunc_idx >= len(base_token_ids) or trunc_idx >= len(cf_token_ids):
                continue
            if trunc_idx in used_trunc_indices:
                continue

            used_trunc_indices.add(trunc_idx)

            src_pos = _resolve_source_value_position(mv)
            if src_pos is None:
                # Skip ambiguous matches instead of emitting incorrect value indices.
                continue
            global_value_index = len(source_values) - 1 - src_pos
            if global_value_index < 0:
                continue

            experiments.append(
                {
                    "experiment_id": f"pair{pair_id}_v{global_value_index}",
                    "pair_id": pair_id,
                    "value_index": global_value_index,
                    "experiment_type": "token_swap_value_match",
                    "truncation_token_index": trunc_idx,
                    "source": {
                        "token_ids": cf_token_ids,
                        "score_token_id": int(cf_token_ids[trunc_idx]),
                    },
                    "target": {
                        "token_ids": base_token_ids,
                        "score_token_id": int(base_token_ids[trunc_idx]),
                    },
                }
            )

    return experiments


def load_experiments_from_input(path: Path) -> List[Dict]:
    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and isinstance(payload.get("experiments"), list):
        return payload["experiments"]

    if isinstance(payload, dict) and isinstance(payload.get("pairs"), list):
        return build_experiments_from_pairs(payload)

    raise ValueError("Input JSON must contain top-level 'experiments' or 'pairs'")


def load_model_and_tokenizer():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=DTYPE,
    )
    model.eval()

    print(f"  Model device: {next(model.parameters()).device}")
    print(f"  Model dtype: {next(model.parameters()).dtype}")
    return model, tokenizer


def forward_logits(model, token_ids: List[int]) -> torch.Tensor:
    input_ids = torch.tensor([token_ids], device=DEVICE, dtype=torch.long)
    with torch.inference_mode():
        out = model(input_ids)
    return out.logits[0]


def forward_logits_batch(model, batch_token_ids: List[List[int]]) -> torch.Tensor:
    input_ids = torch.tensor(batch_token_ids, device=DEVICE, dtype=torch.long)
    with torch.inference_mode():
        out = model(input_ids)
    return out.logits


def compute_next_token_logprobs(logits: torch.Tensor, reference_token_ids: List[int], usable_len: int) -> List[float]:
    log_probs = F.log_softmax(logits, dim=-1)
    scores = []
    for i in range(usable_len - 1):
        next_ref_tok = reference_token_ids[i + 1]
        scores.append(float(log_probs[i, next_ref_tok].item()))
    return scores


def score_logprob_at_position(logits: torch.Tensor, position: int, token_id: int) -> float:
    log_probs = F.log_softmax(logits, dim=-1)
    return float(log_probs[position, token_id].item())


def format_token_label(token: str, max_len: int = 14) -> str:
    cleaned = token.replace("\n", "\\n")
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1] + "~"
    return cleaned


def build_x_tick_labels(tokenizer, base_token_ids: List[int], cf_token_ids: List[int], pos_indices: List[int]) -> List[str]:
    labels = []
    for pos in pos_indices:
        b_tok = tokenizer.decode([base_token_ids[pos]])
        c_tok = tokenizer.decode([cf_token_ids[pos]])
        if base_token_ids[pos] == cf_token_ids[pos]:
            labels.append(f"{pos}:{format_token_label(b_tok)}")
        else:
            labels.append(f"{pos}:{format_token_label(b_tok)}->{format_token_label(c_tok)}")
    return labels


def plot_single_heatmap(
    matrix: np.ndarray,
    title: str,
    out_png: Path,
    x_labels: List[str],
    y_labels: List[str],
    cbar_label: str,
    vmax: float,
) -> None:
    n_rows, n_pos = matrix.shape
    fig, ax = plt.subplots(figsize=(max(10, n_pos * 0.2), max(3.0, 1.6 + 0.45 * n_rows)))
    ax.imshow(matrix, aspect="auto", origin="upper", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlabel("Intervened token position")
    ax.set_ylabel("Intervention row")
    ax.set_title(title)

    if x_labels:
        ax.set_xticks(np.arange(n_pos))
        ax.set_xticklabels(x_labels, rotation=90, fontsize=6)

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(y_labels, fontsize=8)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def main():
    global INPUT_JSON
    global OUTPUT_ROOT_DIR
    global OUTPUT_SUMMARY_JSON
    global EXPERIMENT_INDICES
    global EXPERIMENT_IDS
    global TOKEN_POSITIONS_TO_PATCH
    global PATCH_BATCH_SIZE
    global MODEL_PATH
    global TOKENIZER_PATH
    global DEVICE
    global DTYPE
    global SAVE_PLOTS
    global RESUME

    args = build_arg_parser().parse_args()
    INPUT_JSON = args.input_json
    OUTPUT_ROOT_DIR = args.output_root_dir
    OUTPUT_SUMMARY_JSON = args.output_summary_json or (OUTPUT_ROOT_DIR / "patching_summary.json")

    EXPERIMENT_INDICES = parse_int_list(args.experiment_indices)
    EXPERIMENT_IDS = parse_str_list(args.experiment_ids)
    TOKEN_POSITIONS_TO_PATCH = parse_int_list(args.token_positions_to_patch)
    PATCH_BATCH_SIZE = args.patch_batch_size

    MODEL_PATH = args.model_path
    TOKENIZER_PATH = args.tokenizer_path
    DEVICE = args.device
    DTYPE = torch.float16 if args.dtype == "float16" else torch.float32
    SAVE_PLOTS = not args.no_plots
    RESUME = not args.no_resume

    print("\n" + "=" * 100)
    print("TOKEN-LEVEL COUNTERFACTUAL PATCHING ENGINE")
    print("=" * 100)

    print("\nConfiguration:")
    print(f"  Input JSON: {INPUT_JSON}")
    print(f"  Output root dir: {OUTPUT_ROOT_DIR}")
    print(f"  Summary JSON: {OUTPUT_SUMMARY_JSON}")
    print(f"  Experiment IDs filter: {EXPERIMENT_IDS if EXPERIMENT_IDS else 'none'}")
    print(f"  Token positions to patch: {TOKEN_POSITIONS_TO_PATCH if TOKEN_POSITIONS_TO_PATCH else 'all'}")
    print(f"  Patch batch size: {PATCH_BATCH_SIZE}")
    print(f"  Device: {DEVICE}, Dtype: {DTYPE}")
    print(f"  Save plots: {SAVE_PLOTS}")
    print(f"  Resume: {RESUME}")

    if args.layers_to_patch is not None:
        print("  Note: --layers-to-patch is ignored in token-only mode.")
    if args.layer_stride != 1:
        print("  Note: --layer-stride is ignored in token-only mode.")
    if args.patch_scope not in {"token", "token_all_layers", "cell"}:
        print(f"  Note: unknown --patch-scope '{args.patch_scope}' ignored in token-only mode.")

    model, tokenizer = load_model_and_tokenizer()

    print(f"\nLoading experiments from {INPUT_JSON}...")
    experiments = load_experiments_from_input(INPUT_JSON)
    print(f"Loaded {len(experiments)} experiments")

    if EXPERIMENT_IDS is not None:
        id_to_index = {exp.get("experiment_id", f"idx_{i}"): i for i, exp in enumerate(experiments)}
        exp_indices = []
        for exp_id in EXPERIMENT_IDS:
            if exp_id not in id_to_index:
                print(f"  Warning: experiment_id '{exp_id}' not found. Skipping.")
                continue
            exp_indices.append(id_to_index[exp_id])
    elif EXPERIMENT_INDICES is None:
        exp_indices = list(range(len(experiments)))
    else:
        exp_indices = EXPERIMENT_INDICES

    print(f"Running {len(exp_indices)} experiments...")

    run_records = []
    pair_merged_payloads = {}
    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    existing_run_records: List[Dict] = []
    completed_experiment_ids = set()
    if RESUME:
        pair_merged_payloads = load_existing_pair_payloads(OUTPUT_ROOT_DIR)
        completed_experiment_ids = collect_completed_experiment_ids(pair_merged_payloads)

        if OUTPUT_SUMMARY_JSON.exists():
            try:
                with open(OUTPUT_SUMMARY_JSON, "r") as f:
                    prior_summary = json.load(f)
                if isinstance(prior_summary, dict) and isinstance(prior_summary.get("runs"), list):
                    existing_run_records = prior_summary["runs"]
            except Exception as e:
                print(f"  Warning: failed to load existing summary {OUTPUT_SUMMARY_JSON}: {e}")

        print(f"Resume state: {len(completed_experiment_ids)} experiments already completed")

    for exp_idx in exp_indices:
        if exp_idx >= len(experiments):
            print(f"  Warning: experiment index {exp_idx} out of range. Skipping.")
            continue

        exp = experiments[exp_idx]
        exp_id = exp.get("experiment_id", f"idx_{exp_idx}")

        if RESUME and exp_id in completed_experiment_ids:
            print(f"\n  [{exp_idx}] {exp_id} | already completed, skipping")
            continue

        pair_id = exp["pair_id"]
        value_index = exp.get("value_index", -1)
        exp_type = exp.get("experiment_type", "token_swap")
        trunc_idx = exp["truncation_token_index"]

        print(f"\n  [{exp_idx}] {exp_id} | Pair {pair_id}, Type {exp_type}, Truncation {trunc_idx}")

        cf_token_ids = exp["source"]["token_ids"]
        base_token_ids = exp["target"]["token_ids"]

        usable_len = min(len(cf_token_ids), len(base_token_ids))
        if usable_len < 1:
            print("    Warning: usable token length < 1, skipping.")
            continue

        cf_token_ids = cf_token_ids[:usable_len]
        base_token_ids = base_token_ids[:usable_len]

        if TOKEN_POSITIONS_TO_PATCH is None:
            pos_indices = list(range(trunc_idx))
        else:
            pos_indices = [p for p in TOKEN_POSITIONS_TO_PATCH if 0 <= p < trunc_idx]

        if not pos_indices:
            print(f"    Warning: no valid token positions to patch (truncation at {trunc_idx}), skipping.")
            continue

        cf_score_token_id = exp["source"].get("score_token_id")
        base_score_token_id = exp["target"].get("score_token_id")
        if cf_score_token_id is None or base_score_token_id is None:
            print("    Warning: missing score_token_id in experiment data. Skipping.")
            continue

        print(f"    BASE/CF usable length: {usable_len}")
        print(f"    Positions patched: {len(pos_indices)}")

        print("    Phase 1: baseline BASE forward and scored-token logprobs")
        baseline_logits = forward_logits(model, base_token_ids)
        baseline_cf_scores = compute_next_token_logprobs(baseline_logits, cf_token_ids, usable_len)
        baseline_base_scores = compute_next_token_logprobs(baseline_logits, base_token_ids, usable_len)

        if not isinstance(trunc_idx, int) or trunc_idx <= 0 or trunc_idx >= usable_len:
            print(f"    Warning: invalid truncation index {trunc_idx} for usable_len={usable_len}, skipping.")
            continue

        score_pos = trunc_idx - 1
        scored_token_index = trunc_idx
        baseline_cf_score = score_logprob_at_position(baseline_logits, score_pos, int(cf_score_token_id))
        baseline_base_score = score_logprob_at_position(baseline_logits, score_pos, int(base_score_token_id))

        source_delta_matrix = np.zeros((1, len(pos_indices)), dtype=np.float32)
        base_delta_matrix = np.zeros((1, len(pos_indices)), dtype=np.float32)

        changed_positions = [p for p in pos_indices if base_token_ids[p] != cf_token_ids[p]]
        pos_to_col = {p: i for i, p in enumerate(pos_indices)}

        print("    Phase 2: token-level counterfactual swaps")
        print(f"      Changed positions requiring forwards: {len(changed_positions)} / {len(pos_indices)}")

        for batch_start in range(0, len(changed_positions), PATCH_BATCH_SIZE):
            batch_positions = changed_positions[batch_start: batch_start + PATCH_BATCH_SIZE]
            batch_token_ids: List[List[int]] = []
            for pos in batch_positions:
                patched = list(base_token_ids)
                patched[pos] = cf_token_ids[pos]
                batch_token_ids.append(patched)

            patched_logits = forward_logits_batch(model, batch_token_ids)
            for batch_offset, pos in enumerate(batch_positions):
                col = pos_to_col[pos]
                patched_cf_score = score_logprob_at_position(patched_logits[batch_offset], score_pos, int(cf_score_token_id))
                patched_base_score = score_logprob_at_position(patched_logits[batch_offset], score_pos, int(base_score_token_id))
                source_delta_matrix[0, col] = patched_cf_score - baseline_cf_score
                base_delta_matrix[0, col] = patched_base_score - baseline_base_score

        diff_delta_matrix = source_delta_matrix - base_delta_matrix

        exp_dir = OUTPUT_ROOT_DIR / f"pair{pair_id}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        value_suffix = f"_v{value_index}" if value_index >= 0 else "_tail"
        source_heatmap_png = exp_dir / f"source_heatmap{value_suffix}_t{trunc_idx}.png"
        base_heatmap_png = exp_dir / f"base_heatmap{value_suffix}_t{trunc_idx}.png"
        diff_heatmap_png = exp_dir / f"diff_heatmap{value_suffix}_t{trunc_idx}.png"
        pair_json = exp_dir / "pair_matrices.json"

        x_tick_labels = build_x_tick_labels(tokenizer, base_token_ids, cf_token_ids, pos_indices)

        all_abs = np.concatenate(
            [
                np.abs(source_delta_matrix[np.isfinite(source_delta_matrix)]),
                np.abs(base_delta_matrix[np.isfinite(base_delta_matrix)]),
                np.abs(diff_delta_matrix[np.isfinite(diff_delta_matrix)]),
            ]
        )
        shared_vmax = float(all_abs.max()) if all_abs.size > 0 else 1.0
        if shared_vmax <= 0:
            shared_vmax = 1.0

        diff_abs = np.abs(diff_delta_matrix[np.isfinite(diff_delta_matrix)])
        diff_vmax = float(diff_abs.max()) if diff_abs.size > 0 else 1.0
        if diff_vmax <= 0:
            diff_vmax = 1.0

        source_title = (
            f"{exp_id} | pair {pair_id} | type {exp_type} | trunc {trunc_idx}\n"
            f"CF-swap delta: log P(cf token@{scored_token_index})"
        )
        base_title = (
            f"{exp_id} | pair {pair_id} | type {exp_type} | trunc {trunc_idx}\n"
            f"CF-swap delta: log P(base token@{scored_token_index})"
        )
        diff_title = (
            f"{exp_id} | pair {pair_id} | type {exp_type} | trunc {trunc_idx}\n"
            f"DIFF delta: cf - base at token@{scored_token_index}"
        )

        if SAVE_PLOTS:
            y_labels = ["token-swap"]
            plot_single_heatmap(
                matrix=source_delta_matrix,
                title=source_title,
                out_png=source_heatmap_png,
                x_labels=x_tick_labels,
                y_labels=y_labels,
                cbar_label="Delta log P(cf token)",
                vmax=shared_vmax,
            )
            plot_single_heatmap(
                matrix=base_delta_matrix,
                title=base_title,
                out_png=base_heatmap_png,
                x_labels=x_tick_labels,
                y_labels=y_labels,
                cbar_label="Delta log P(base token)",
                vmax=shared_vmax,
            )
            plot_single_heatmap(
                matrix=diff_delta_matrix,
                title=diff_title,
                out_png=diff_heatmap_png,
                x_labels=x_tick_labels,
                y_labels=y_labels,
                cbar_label="Delta log P(cf token) - Delta log P(base token)",
                vmax=diff_vmax,
            )

        matrix_payload = {
            "experiment_id": exp_id,
            "pair_id": pair_id,
            "value_index": value_index,
            "experiment_type": exp_type,
            "truncation_token_index": trunc_idx,
            "usable_len": usable_len,
            "layer_indices": [-1],
            "patched_layer_indices": [],
            "position_indices": pos_indices,
            "score_target_definition": "token at truncation index n (predicted from position n-1)",
            "score_position_used": score_pos,
            "scored_token_index": scored_token_index,
            "scored_source_token_id": int(cf_score_token_id),
            "scored_base_token_id": int(base_score_token_id),
            "scored_source_token_text": tokenizer.decode([int(cf_score_token_id)]),
            "scored_base_token_text": tokenizer.decode([int(base_score_token_id)]),
            "intervention_mode": "counterfactual_token_swap",
            "baseline_prompt": "base_trace",
            "swapped_prompt": "base_with_single_cf_token",
            "n_changed_positions": len(changed_positions),
            "patch_batch_size": PATCH_BATCH_SIZE,
            "baseline_scored_source_logprob": baseline_cf_score,
            "baseline_scored_base_logprob": baseline_base_score,
            "x_tick_labels": x_tick_labels,
            "baseline_next_source_logprobs": baseline_cf_scores,
            "baseline_next_base_logprobs": baseline_base_scores,
            "source_delta_matrix": source_delta_matrix.tolist(),
            "base_delta_matrix": base_delta_matrix.tolist(),
            "diff_delta_matrix": diff_delta_matrix.tolist(),
        }

        pair_key = f"pair{pair_id}"
        if pair_key not in pair_merged_payloads:
            pair_merged_payloads[pair_key] = {
                "schema_version": "v1",
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "model": "Qwen2.5-32B",
                "pair_id": pair_id,
                "entries": [],
            }

        old_entries = pair_merged_payloads[pair_key].get("entries", [])
        def _same_trunc(entry: Dict) -> bool:
            try:
                return int(entry.get("truncation_token_index", -1)) == int(trunc_idx)
            except Exception:
                return False

        def _same_value_index(entry: Dict) -> bool:
            try:
                return int(entry.get("value_index", -999999)) == int(value_index)
            except Exception:
                return False

        # Replace prior entries by experiment id, value index, or truncation index.
        # This keeps resume robust across older schema variants that used local value indices.
        new_entries = [
            e for e in old_entries
            if e.get("experiment_id") != exp_id
            and not _same_trunc(e)
            and not _same_value_index(e)
        ]
        new_entries.append(matrix_payload)
        pair_merged_payloads[pair_key]["entries"] = new_entries
        completed_experiment_ids.add(exp_id)

        running_payload = pair_merged_payloads[pair_key]
        running_entries = running_payload.get("entries", [])
        running_entries.sort(key=lambda x: x.get("truncation_token_index", 0), reverse=True)
        running_payload["n_entries"] = len(running_entries)
        with open(pair_json, "w") as f:
            json.dump(running_payload, f, indent=2)

        run_records.append(
            {
                "experiment_id": exp_id,
                "pair_id": pair_id,
                "value_index": value_index,
                "experiment_type": exp_type,
                "truncation_token_index": trunc_idx,
                "source_heatmap_png": str(source_heatmap_png) if SAVE_PLOTS else None,
                "base_heatmap_png": str(base_heatmap_png) if SAVE_PLOTS else None,
                "diff_heatmap_png": str(diff_heatmap_png) if SAVE_PLOTS else None,
                "pair_json": str(pair_json),
                "usable_len": usable_len,
                "n_layers": 0,
                "n_matrix_rows": int(source_delta_matrix.shape[0]),
                "n_positions": len(pos_indices),
                "n_changed_positions": len(changed_positions),
                "intervention_mode": "counterfactual_token_swap",
            }
        )

        if SAVE_PLOTS:
            print(f"    Saved: {source_heatmap_png}")
            print(f"    Saved: {base_heatmap_png}")
            print(f"    Saved: {diff_heatmap_png}")

    for pair_key, payload in pair_merged_payloads.items():
        pair_dir = OUTPUT_ROOT_DIR / pair_key
        pair_json = pair_dir / "pair_matrices.json"
        entries = payload.get("entries", [])
        entries.sort(key=lambda x: x.get("truncation_token_index", 0), reverse=True)
        payload["n_entries"] = len(entries)
        with open(pair_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved merged pair JSON: {pair_json}")

    print("\n" + "=" * 100)
    print(f"Writing summary to {OUTPUT_SUMMARY_JSON}...")

    merged_runs = merge_run_records(existing_run_records, run_records) if RESUME else run_records

    output = {
        "schema_version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": "Qwen2.5-32B",
        "total_runs": len(merged_runs),
        "runs_completed_this_invocation": len(run_records),
        "configuration": {
            "experiment_ids_filter": EXPERIMENT_IDS,
            "experiment_indices_filter": EXPERIMENT_INDICES,
            "positions_to_patch": TOKEN_POSITIONS_TO_PATCH if TOKEN_POSITIONS_TO_PATCH else "all",
            "score_target_definition": "token at truncation index n (predicted from position n-1)",
            "resume": RESUME,
            "patch_batch_size": PATCH_BATCH_SIZE,
            "intervention_mode": "counterfactual_token_swap",
        },
        "runs": merged_runs,
    }

    OUTPUT_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SUMMARY_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(merged_runs)} total run records ({len(run_records)} newly completed)")
    print(f"✓ Summary: {OUTPUT_SUMMARY_JSON}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
