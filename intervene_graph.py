"""Causal activation patching runner.

For each experiment entry from truncated_traces.json (already tied to a truncation index):
1. Run SOURCE forward pass and capture all layer activations.
2. Run TARGET forward pass and score log P(next SOURCE token) at each position.
3. For every layer and token position, patch SOURCE->TARGET at that single cell,
     rerun TARGET, and measure delta logprob.
4. Save one heatmap per truncation entry.

Outputs are organized as:
    OUTPUT_ROOT_DIR/
        <experiment_id>/
            source_heatmap_t<trunc_idx>.png
            base_heatmap_t<trunc_idx>.png
            sum_heatmap_t<trunc_idx>.png
            matrix_t<trunc_idx>.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# GLOBAL CONFIGURATION - EDIT THESE
# ============================================================================

# Path to truncated traces from inspect_tokens_simple.py
TRUNCATED_TRACES_JSON = Path("/home/wuroderi/links/scratch/reasoning_traces/Qwen2.5-32B/velocity/truncated_traces.json")

# Output directory (contains per-experiment folders)
OUTPUT_ROOT_DIR = Path("/home/wuroderi/links/scratch/reasoning_traces/Qwen2.5-32B/velocity/patch_runs")

# Summary JSON across all runs
OUTPUT_SUMMARY_JSON = OUTPUT_ROOT_DIR / "patching_summary.json"

# Which experiments to run (list of indices from truncated_traces.json, or None for all)
EXPERIMENT_INDICES = None  # None = run all, or [0, 1, 2] for specific experiments

# Optional: select by experiment_id strings (e.g., ["p0_cot0", "p1_cot1"]).
# If set, this takes precedence over EXPERIMENT_INDICES.
EXPERIMENT_IDS = None

# Optional layer filter (None means all layers)
LAYERS_TO_PATCH = None

# Optional token position filter (None means all valid positions 0..seq_len-2)
TOKEN_POSITIONS_TO_PATCH = None

# Device for computation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

MODEL_PATH = "/home/wuroderi/links/projects/def-rgrosse/wuroderi/models/Qwen2.5-32B"
TOKENIZER_PATH = MODEL_PATH

def load_model_and_tokenizer():
    """Load model and tokenizer."""
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


def get_transformer_layers(model):
    """Return an indexable list-like transformer block container."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Could not locate transformer layers on model.")


def forward_logits(model, token_ids: List[int]) -> torch.Tensor:
    """Forward pass returning logits [seq_len, vocab]."""
    input_ids = torch.tensor([token_ids], device=DEVICE, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids)
    return out.logits[0]


def compute_next_token_logprobs(logits: torch.Tensor, reference_token_ids: List[int], usable_len: int) -> List[float]:
    """Score log P(reference[i+1]) from logits at position i, i in [0, usable_len-2]."""
    log_probs = F.log_softmax(logits, dim=-1)
    scores = []
    for i in range(usable_len - 1):
        next_ref_tok = reference_token_ids[i + 1]
        scores.append(float(log_probs[i, next_ref_tok].item()))
    return scores


def score_logprob_at_position(logits: torch.Tensor, position: int, token_id: int) -> float:
    """Return log P(token_id) from logits at a specific position."""
    log_probs = F.log_softmax(logits, dim=-1)
    return float(log_probs[position, token_id].item())


def capture_source_activations(model, layers, source_token_ids: List[int], layer_indices: List[int]) -> Dict[int, torch.Tensor]:
    """Capture full hidden states for SOURCE at chosen layers."""
    source_acts: Dict[int, torch.Tensor] = {}

    def make_capture_hook(layer_idx: int):
        def _hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            source_acts[layer_idx] = hidden.detach().clone()
        return _hook

    hooks = []
    for lidx in layer_indices:
        hooks.append(layers[lidx].register_forward_hook(make_capture_hook(lidx)))

    _ = forward_logits(model, source_token_ids)

    for h in hooks:
        h.remove()

    return source_acts


def patched_logits_single_cell(
    model,
    layers,
    target_token_ids: List[int],
    source_acts: Dict[int, torch.Tensor],
    layer_idx: int,
    token_pos: int,
) -> torch.Tensor:
    """Forward TARGET with one patch cell: (layer_idx, token_pos)."""

    def _patch_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        src_hidden = source_acts[layer_idx].to(patched.device)
        if token_pos < patched.shape[1] and token_pos < src_hidden.shape[1]:
            patched[:, token_pos, :] = src_hidden[:, token_pos, :]
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    hook = layers[layer_idx].register_forward_hook(_patch_hook)
    try:
        logits = forward_logits(model, target_token_ids)
    finally:
        hook.remove()
    return logits


def format_token_label(token: str, max_len: int = 14) -> str:
    """Short, readable token label for axis tick text."""
    cleaned = token.replace("\n", "\\n")
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1] + "~"
    return cleaned


def build_x_tick_labels(tokenizer, source_token_ids: List[int], pos_indices: List[int]) -> List[str]:
    """Build labels like '12:tok' for each patched position using SOURCE token at that position."""
    labels = []
    for pos in pos_indices:
        tok = tokenizer.decode([source_token_ids[pos]])
        labels.append(f"{pos}:{format_token_label(tok)}")
    return labels


def plot_single_heatmap(
    matrix: np.ndarray,
    title: str,
    out_png: Path,
    x_labels: List[str],
    layer_indices: List[int],
    cbar_label: str,
    vmax: float,
) -> None:
    """Save one standard heatmap."""
    n_layers, n_pos = matrix.shape
    fig, ax = plt.subplots(figsize=(max(10, n_pos * 0.2), 8))
    ax.imshow(matrix, aspect="auto", origin="upper", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlabel("Patched token position : source token")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    if x_labels:
        ax.set_xticks(np.arange(n_pos))
        ax.set_xticklabels(x_labels, rotation=90, fontsize=6)

    y_tick_step = max(1, n_layers // 16)
    y_ticks = np.arange(0, n_layers, y_tick_step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(layer_indices[i]) for i in y_ticks], fontsize=8)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def main():
    print("\n" + "=" * 100)
    print("ACTIVATION PATCHING ENGINE")
    print("=" * 100)

    print(f"\nConfiguration:")
    print(f"  Truncated traces: {TRUNCATED_TRACES_JSON}")
    print(f"  Output root dir: {OUTPUT_ROOT_DIR}")
    print(f"  Summary JSON: {OUTPUT_SUMMARY_JSON}")
    print(f"  Experiment IDs filter: {EXPERIMENT_IDS if EXPERIMENT_IDS else 'none'}")
    print(f"  Layers to patch: {LAYERS_TO_PATCH if LAYERS_TO_PATCH else 'all'}")
    print(f"  Token positions to patch: {TOKEN_POSITIONS_TO_PATCH if TOKEN_POSITIONS_TO_PATCH else 'all'}")
    print(f"  Device: {DEVICE}, Dtype: {DTYPE}")

    model, tokenizer = load_model_and_tokenizer()
    _ = tokenizer
    layers = get_transformer_layers(model)
    n_layers_total = len(layers)
    print(f"  Resolved transformer layers: {n_layers_total}")

    print(f"\nLoading truncated traces from {TRUNCATED_TRACES_JSON}...")
    with open(TRUNCATED_TRACES_JSON, "r") as f:
        traces_data = json.load(f)

    experiments = traces_data["experiments"]
    print(f"Loaded {len(experiments)} experiments")

    if EXPERIMENT_IDS is not None:
        id_to_index = {
            exp.get("experiment_id", f"idx_{i}"): i for i, exp in enumerate(experiments)
        }
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
    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    for exp_idx in exp_indices:
        if exp_idx >= len(experiments):
            print(f"  Warning: experiment index {exp_idx} out of range. Skipping.")
            continue

        exp = experiments[exp_idx]
        exp_id = exp.get("experiment_id", f"idx_{exp_idx}")
        pair_id = exp["pair_id"]
        exp_type = exp["experiment_type"]
        trunc_idx = exp["truncation_token_index"]

        print(f"\n  [{exp_idx}] {exp_id} | Pair {pair_id}, Type {exp_type}, Truncation {trunc_idx}")

        source_token_ids = exp["source"]["token_ids"]
        base_token_ids = exp["target"]["token_ids"]

        usable_len = min(len(source_token_ids), len(base_token_ids))
        if usable_len < 1:
            print("    Warning: usable token length < 1, skipping.")
            continue

        source_token_ids = source_token_ids[:usable_len]
        base_token_ids = base_token_ids[:usable_len]

        layer_indices = list(range(n_layers_total)) if LAYERS_TO_PATCH is None else list(LAYERS_TO_PATCH)
        layer_indices = [l for l in layer_indices if 0 <= l < n_layers_total]
        if not layer_indices:
            print("    Warning: no valid layers to patch, skipping.")
            continue

        pos_indices = list(range(usable_len)) if TOKEN_POSITIONS_TO_PATCH is None else list(TOKEN_POSITIONS_TO_PATCH)
        pos_indices = [p for p in pos_indices if 0 <= p < usable_len]
        if not pos_indices:
            print("    Warning: no valid token positions to patch, skipping.")
            continue

        source_score_token_id = exp["source"].get("score_token_id")
        base_score_token_id = exp["target"].get("score_token_id")
        if source_score_token_id is None or base_score_token_id is None:
            print("    Warning: missing score_token_id in experiment data. Regenerate truncated_traces.json with inspect_tokens_simple.py. Skipping.")
            continue

        print(f"    SOURCE/TARGET usable length: {usable_len}")
        print(f"    Layers patched: {len(layer_indices)} | Positions patched: {len(pos_indices)}")

        print("    Phase 1: capture SOURCE activations (all selected layers)")
        source_acts = capture_source_activations(model, layers, source_token_ids, layer_indices)

        print("    Phase 2: baseline TARGET forward and next-SOURCE-token logprobs")
        baseline_logits = forward_logits(model, base_token_ids)
        baseline_source_scores = compute_next_token_logprobs(baseline_logits, source_token_ids, usable_len)
        baseline_base_scores = compute_next_token_logprobs(baseline_logits, base_token_ids, usable_len)
        # Truncation index n is exclusive context [0..n-1]; score token index n
        # from logits at position n-1.
        score_pos = usable_len - 1
        scored_token_index = usable_len
        baseline_source_score = score_logprob_at_position(baseline_logits, score_pos, int(source_score_token_id))
        baseline_base_score = score_logprob_at_position(baseline_logits, score_pos, int(base_score_token_id))

        source_delta_matrix = np.full((len(layer_indices), len(pos_indices)), np.nan, dtype=np.float32)
        base_delta_matrix = np.full((len(layer_indices), len(pos_indices)), np.nan, dtype=np.float32)

        print("    Phase 3: single-cell patching over (layer, position)")
        for li, layer_idx in enumerate(layer_indices):
            for pi, pos in enumerate(pos_indices):
                patched_logits = patched_logits_single_cell(
                    model=model,
                    layers=layers,
                    target_token_ids=base_token_ids,
                    source_acts=source_acts,
                    layer_idx=layer_idx,
                    token_pos=pos,
                )
                patched_source_score = score_logprob_at_position(patched_logits, score_pos, int(source_score_token_id))
                patched_base_score = score_logprob_at_position(patched_logits, score_pos, int(base_score_token_id))
                source_delta_matrix[li, pi] = patched_source_score - baseline_source_score
                base_delta_matrix[li, pi] = patched_base_score - baseline_base_score

        exp_dir = OUTPUT_ROOT_DIR / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        source_heatmap_png = exp_dir / f"source_heatmap_t{trunc_idx}.png"
        base_heatmap_png = exp_dir / f"base_heatmap_t{trunc_idx}.png"
        diff_heatmap_png = exp_dir / f"diff_heatmap_t{trunc_idx}.png"
        matrix_json = exp_dir / f"matrix_t{trunc_idx}.json"
        x_tick_labels = build_x_tick_labels(tokenizer, source_token_ids, pos_indices)
        diff_delta_matrix = source_delta_matrix - base_delta_matrix

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

        source_title = (
            f"{exp_id} | pair {pair_id} | type {exp_type} | trunc {trunc_idx}\n"
            f"SOURCE delta: log P(source token@{scored_token_index})"
        )
        base_title = (
            f"{exp_id} | pair {pair_id} | type {exp_type} | trunc {trunc_idx}\n"
            f"BASE delta: log P(base token@{scored_token_index})"
        )
        diff_title = (
            f"{exp_id} | pair {pair_id} | type {exp_type} | trunc {trunc_idx}\n"
            f"DIFF delta: source - base at token@{scored_token_index}"
        )

        plot_single_heatmap(
            matrix=source_delta_matrix,
            title=source_title,
            out_png=source_heatmap_png,
            x_labels=x_tick_labels,
            layer_indices=layer_indices,
            cbar_label="Delta log P(source token)",
            vmax=shared_vmax,
        )
        plot_single_heatmap(
            matrix=base_delta_matrix,
            title=base_title,
            out_png=base_heatmap_png,
            x_labels=x_tick_labels,
            layer_indices=layer_indices,
            cbar_label="Delta log P(base token)",
            vmax=shared_vmax,
        )
        plot_single_heatmap(
            matrix=diff_delta_matrix,
            title=diff_title,
            out_png=diff_heatmap_png,
            x_labels=x_tick_labels,
            layer_indices=layer_indices,
            cbar_label="Delta log P(source token) - Delta log P(base token)",
            vmax=shared_vmax,
        )

        matrix_payload = {
            "experiment_id": exp_id,
            "pair_id": pair_id,
            "experiment_type": exp_type,
            "truncation_token_index": trunc_idx,
            "usable_len": usable_len,
            "layer_indices": layer_indices,
            "position_indices": pos_indices,
            "score_target_definition": "token at truncation index n (predicted from position n-1)",
            "score_position_used": score_pos,
            "scored_token_index": scored_token_index,
            "scored_source_token_id": int(source_score_token_id),
            "scored_base_token_id": int(base_score_token_id),
            "scored_source_token_text": tokenizer.decode([int(source_score_token_id)]),
            "scored_base_token_text": tokenizer.decode([int(base_score_token_id)]),
            "baseline_scored_source_logprob": baseline_source_score,
            "baseline_scored_base_logprob": baseline_base_score,
            "x_tick_labels": x_tick_labels,
            "baseline_next_source_logprobs": baseline_source_scores,
            "baseline_next_base_logprobs": baseline_base_scores,
            "source_delta_matrix": source_delta_matrix.tolist(),
            "base_delta_matrix": base_delta_matrix.tolist(),
            "diff_delta_matrix": diff_delta_matrix.tolist(),
        }
        with open(matrix_json, "w") as f:
            json.dump(matrix_payload, f, indent=2)

        run_records.append(
            {
                "experiment_id": exp_id,
                "pair_id": pair_id,
                "experiment_type": exp_type,
                "truncation_token_index": trunc_idx,
                "source_heatmap_png": str(source_heatmap_png),
                "base_heatmap_png": str(base_heatmap_png),
                "diff_heatmap_png": str(diff_heatmap_png),
                "matrix_json": str(matrix_json),
                "usable_len": usable_len,
                "n_layers": len(layer_indices),
                "n_positions": len(pos_indices),
            }
        )
        print(f"    Saved: {source_heatmap_png}")
        print(f"    Saved: {base_heatmap_png}")
        print(f"    Saved: {diff_heatmap_png}")

    print(f"\n" + "=" * 100)
    print(f"Writing summary to {OUTPUT_SUMMARY_JSON}...")

    output = {
        "schema_version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": "Qwen2.5-32B",
        "total_runs": len(run_records),
        "configuration": {
            "experiment_ids_filter": EXPERIMENT_IDS,
            "experiment_indices_filter": EXPERIMENT_INDICES,
            "layers_to_patch": LAYERS_TO_PATCH if LAYERS_TO_PATCH else "all",
            "positions_to_patch": TOKEN_POSITIONS_TO_PATCH if TOKEN_POSITIONS_TO_PATCH else "all",
            "score_target_definition": "token at truncation index n (predicted from position n-1)",
        },
        "runs": run_records,
    }

    OUTPUT_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SUMMARY_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(run_records)} run records")
    print(f"✓ Summary: {OUTPUT_SUMMARY_JSON}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
