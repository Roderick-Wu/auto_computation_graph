"""
Token inspection tool for activation patching experiments.

Each pair_id generates 2 patching experiments:
  - Experiment A: trace_0.original ← trace_0.with_1_values (swap trace_1 values into trace_0)
  - Experiment B: trace_1.original ← trace_1.with_0_values (swap trace_0 values into trace_1)

With 3 pair_ids, you have 6 total experiments (3 pairs × 2 directions).

This script:
1. Prints all token indices with their tokens for visualization
2. Allows specifying multiple truncation token indices per pair
3. Generates truncated_traces.json for intervene_graph.py

GLOBAL CONFIGURATION:
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoTokenizer

# ============================================================================
# GLOBAL CONFIGURATION - EDIT THESE
# ============================================================================

# Which pair_ids to inspect from prompt_pairs_top3.json
PAIR_IDS = [0, 1, 2]

# For each expanded experiment_id, specify truncation TOKEN indices.
# Expansion rule per pair_id:
#   p{pair_id}_cot0 : trace_0.original  -> trace_0.with_1_values
#   p{pair_id}_cot1 : trace_1.original  -> trace_1.with_0_values
#
# This lets you set different truncation points for each CoT direction.
TRUNCATION_TOKEN_INDICES_BY_EXPERIMENT = {
    "p0_cot0": [90, 82, 77, 65, 52],
    "p0_cot1": [122, 114, 110, 102, 87, 78, 70, 61, 44],
    "p1_cot0": [95, 88, 84, 72, 59, 50],
    "p1_cot1": [140, 132, 126, 113, 94, 77, 69, 52],
    "p2_cot0": [91, 84, 80, 70, 60, 44],
    "p2_cot1": [95, 88, 84, 72, 59, 50],
}

# ============================================================================
# CONSTANTS
# ============================================================================

MODEL_PATH = "/home/wuroderi/links/projects/def-rgrosse/wuroderi/models/Qwen2.5-32B"
PAIRS_JSON = Path("/home/wuroderi/links/scratch/reasoning_traces/Qwen2.5-32B/velocity/prompt_pairs_top3.json")
OUTPUT_JSON = Path("/home/wuroderi/links/scratch/reasoning_traces/Qwen2.5-32B/velocity/truncated_traces.json")

# ============================================================================
# HELPERS
# ============================================================================


def tokenize_text(tokenizer, text: str) -> dict:
    """Tokenize text and return tokens, token_ids, and offsets."""
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    ids = enc["input_ids"][0].tolist()
    offsets = enc["offset_mapping"][0].tolist()
    
    tokens = []
    for i, token_id in enumerate(ids):
        token_str = tokenizer.decode([token_id]).replace("\n", "\\n")
        tokens.append(token_str)
    
    return {
        "tokens": tokens,
        "token_ids": ids,
        "offsets": offsets,
    }


def truncate_tokens(tokens_dict: dict, token_index: int) -> dict:
    """Truncate token dict to a specific token index (exclusive)."""
    truncated = {
        "tokens": tokens_dict["tokens"][:token_index],
        "token_ids": tokens_dict["token_ids"][:token_index],
        "offsets": tokens_dict["offsets"][:token_index],
    }
    return truncated


def reconstruct_text(tokens_dict: dict) -> str:
    """Reconstruct text from truncated token offsets."""
    if not tokens_dict["offsets"]:
        return ""
    first_offset_start = tokens_dict["offsets"][0][0]
    last_offset_end = tokens_dict["offsets"][-1][1]
    # This is approximate; ideally we'd have the original text to slice
    return " ".join(tokens_dict["tokens"])


def print_tokens_side_by_side(source_tokens: dict, target_tokens: dict, source_label: str, target_label: str) -> None:
    """Print SOURCE and TARGET tokens in one side-by-side table for easy comparison."""
    source_len = len(source_tokens["tokens"])
    target_len = len(target_tokens["tokens"])
    max_len = max(source_len, target_len)

    # Keep columns narrow enough to avoid terminal wrapping.
    left_width = 24
    right_width = 24

    print(f"\n{'='*100}")
    print(f"TOKEN COMPARISON | {source_label} ({source_len}) vs {target_label} ({target_len})")
    print(f"{'='*100}")
    print(
        f"{'idx':<4} {'s_start':<7} {'s_end':<6} {'source_token':<{left_width}}"
        f" || {'idx':<4} {'t_start':<7} {'t_end':<6} {'target_token':<{right_width}}"
    )
    print("-" * 100)

    for i in range(max_len):
        if i < source_len:
            s_start, s_end = source_tokens["offsets"][i]
            s_tok = source_tokens["tokens"][i]
            s_tok_disp = s_tok[: left_width - 1] if len(s_tok) >= left_width else s_tok
            left = f"{i:<4} {s_start:<7} {s_end:<6} {s_tok_disp:<{left_width}}"
        else:
            left = f"{'':<4} {'':<7} {'':<6} {'':<{left_width}}"

        if i < target_len:
            t_start, t_end = target_tokens["offsets"][i]
            t_tok = target_tokens["tokens"][i]
            t_tok_disp = t_tok[: right_width - 1] if len(t_tok) >= right_width else t_tok
            right = f"{i:<4} {t_start:<7} {t_end:<6} {t_tok_disp:<{right_width}}"
        else:
            right = f"{'':<4} {'':<7} {'':<6} {'':<{right_width}}"

        print(f"{left} || {right}")


def build_experiment_specs(pair_ids: list, pairs: dict) -> list:
    """Expand pair_ids into directional experiment specs."""
    specs = []
    for pair_id in pair_ids:
        if pair_id not in pairs:
            continue

        specs.append(
            {
                "experiment_id": f"p{pair_id}_cot0",
                "pair_id": pair_id,
                "experiment_type": "A",
                "source_key": "cot_0",
                "source_field": "original",
                "target_field": "with_1_values",
                "description": "Patch trace_0 (inject trace_1 values)",
            }
        )
        specs.append(
            {
                "experiment_id": f"p{pair_id}_cot1",
                "pair_id": pair_id,
                "experiment_type": "B",
                "source_key": "cot_1",
                "source_field": "original",
                "target_field": "with_0_values",
                "description": "Patch trace_1 (inject trace_0 values)",
            }
        )
    return specs


def main():
    print("\n" + "=" * 100)
    print("ACTIVATION PATCHING EXPERIMENT INSPECTOR")
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  Pair IDs to inspect: {PAIR_IDS}")
    print(f"  Truncation indices by experiment_id: {TRUNCATION_TOKEN_INDICES_BY_EXPERIMENT}")
    print(f"  Output JSON: {OUTPUT_JSON}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Load pairs
    print(f"Loading pairs from {PAIRS_JSON}...")
    with open(PAIRS_JSON, "r") as f:
        payload = json.load(f)
    
    pairs = {pair["pair_id"]: pair for pair in payload["pairs"]}
    print(f"Loaded {len(pairs)} pairs from JSON.\n")

    experiment_specs = build_experiment_specs(PAIR_IDS, pairs)
    print(f"Expanded into {len(experiment_specs)} directional experiment_ids.")
    print("  " + ", ".join(spec["experiment_id"] for spec in experiment_specs))
    
    # Collect truncated experiments
    truncated_experiments = []
    total_experiments = 0
    
    # Process each expanded experiment_id
    for spec in experiment_specs:
        pair_id = spec["pair_id"]
        experiment_id = spec["experiment_id"]

        if pair_id not in pairs:
            print(f"\nERROR: pair_id {pair_id} not found in JSON. Skipping {experiment_id}.")
            continue

        pair = pairs[pair_id]
        trunc_indices = TRUNCATION_TOKEN_INDICES_BY_EXPERIMENT.get(experiment_id, [])
        if not trunc_indices:
            print(f"\nSkipping {experiment_id}: no truncation indices configured.")
            continue

        print("\n" + "#" * 100)
        print(f"# PAIR_ID {pair_id} | EXPERIMENT_ID {experiment_id}")
        print("#" * 100)
        print("Values:")
        print(f"  trace_0: m={pair['values']['trace_0']['m']}, "
              f"ke={pair['values']['trace_0']['ke']}, "
              f"v={pair['values']['trace_0']['v']}, "
              f"d={pair['values']['trace_0']['d']}, "
              f"t={pair['values']['trace_0']['t']}")
        print(f"  trace_1: m={pair['values']['trace_1']['m']}, "
              f"ke={pair['values']['trace_1']['ke']}, "
              f"v={pair['values']['trace_1']['v']}, "
              f"d={pair['values']['trace_1']['d']}, "
              f"t={pair['values']['trace_1']['t']}")

        print(f"\n{'*'*100}")
        print(spec["description"])
        print(f"{'*'*100}")

        source_text = pair[spec["source_key"]][spec["source_field"]]
        target_text = pair[spec["source_key"]][spec["target_field"]]

        print(f"\n--- SOURCE ({spec['source_key']}.{spec['source_field']}) ---")
        print(source_text)
        source_tokens = tokenize_text(tokenizer, source_text)

        print(f"\n--- TARGET ({spec['source_key']}.{spec['target_field']}) ---")
        print(target_text)
        target_tokens = tokenize_text(tokenizer, target_text)

        print_tokens_side_by_side(source_tokens, target_tokens, "tokens_0", "tokens_1")

        # Generate truncated variants for this experiment_id.
        # For truncation index n (exclusive), context is tokens [0..n-1],
        # and score target token is token index n.
        for trunc_idx in trunc_indices:
            if trunc_idx <= 0:
                print(f"  Warning: truncation index {trunc_idx} must be > 0. Skipping.")
                continue
            if trunc_idx >= len(source_tokens["tokens"]) or trunc_idx >= len(target_tokens["tokens"]):
                print(f"  Warning: truncation index {trunc_idx} must be < token count to score token n. Skipping.")
                continue

            source_truncated = truncate_tokens(source_tokens, trunc_idx)
            target_truncated = truncate_tokens(target_tokens, trunc_idx)
            source_score_token_id = source_tokens["token_ids"][trunc_idx]
            target_score_token_id = target_tokens["token_ids"][trunc_idx]
            source_score_token_text = source_tokens["tokens"][trunc_idx]
            target_score_token_text = target_tokens["tokens"][trunc_idx]

            truncated_experiments.append({
                "experiment_id": experiment_id,
                "pair_id": pair_id,
                "experiment_type": spec["experiment_type"],
                "truncation_token_index": trunc_idx,
                "source": {
                    "tokens": source_truncated["tokens"],
                    "token_ids": source_truncated["token_ids"],
                    "truncation_length": trunc_idx,
                    "score_token_id": source_score_token_id,
                    "score_token_text": source_score_token_text,
                },
                "target": {
                    "tokens": target_truncated["tokens"],
                    "token_ids": target_truncated["token_ids"],
                    "truncation_length": trunc_idx,
                    "score_token_id": target_score_token_id,
                    "score_token_text": target_score_token_text,
                },
            })
            total_experiments += 1
    
    # Write output JSON
    print("\n" + "=" * 100)
    print(f"Writing {total_experiments} truncated experiments to {OUTPUT_JSON}...")
    print("=" * 100)
    
    output = {
        "schema_version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": "Qwen2.5-32B",
        "total_experiments": total_experiments,
        "pair_ids": PAIR_IDS,
        "truncation_indices_by_experiment": TRUNCATION_TOKEN_INDICES_BY_EXPERIMENT,
        "experiments": truncated_experiments,
    }
    
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Wrote {total_experiments} experiments to {OUTPUT_JSON}")
    print("\nNext step: Use intervene_graph.py to run activation patching on these experiments.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
