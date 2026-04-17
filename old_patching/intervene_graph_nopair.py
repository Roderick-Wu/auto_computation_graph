"""Causal activation noise-patching runner.

For each experiment entry built from a base trace:
1. Run the clean prompt forward pass and capture all layer activations.
2. Score the true next token at the truncation point.
3. Patch activations with noise and measure delta logprob:
   - `cell`: for every layer and token position, patch one cell.
   - `token_all_layers`: for every token position, patch that token cell at
     all selected layers in one pass.
4. Save one heatmap and JSON payload per numeric value in the CoT.

Outputs are organized as:
    OUTPUT_ROOT_DIR/
        pair0/
            noise_heatmap_v0_t<trunc_idx>.png
            pair_matrices.json
        pair1/
            ...
"""

import argparse
import hashlib
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
# GLOBAL CONFIGURATION - EDIT THESE
# ============================================================================

# Input can be one of:
#  - legacy truncated experiments payload with top-level "experiments"
#  - reject traces payload with top-level "traces"
#  - aligned pairs payload with top-level "pairs"
INPUT_JSON = resolve_auto_traces_root("Qwen2.5-32B", "velocity") / "aligned_pairs.json"

# Output directory (contains per-experiment folders)
OUTPUT_ROOT_DIR = resolve_auto_traces_root("Qwen2.5-32B", "velocity") / "patch_solo"

# Summary JSON across all runs
OUTPUT_SUMMARY_JSON = OUTPUT_ROOT_DIR / "patching_summary.json"

# Which experiments to run (list of indices from truncated_traces.json, or None for all)
EXPERIMENT_INDICES = None  # None = run all, or [0, 1, 2] for specific experiments

# Optional: select by experiment_id strings (e.g., ["p0_cot0", "p1_cot1"]).
# If set, this takes precedence over EXPERIMENT_INDICES.
EXPERIMENT_IDS = None

# Optional layer filter (None means all layers)
LAYERS_TO_PATCH = None

# Patch every Nth layer (1 = all layers).
LAYER_STRIDE = 1

# Optional token position filter (None means all valid positions 0..seq_len-2)
TOKEN_POSITIONS_TO_PATCH = None

# Number of token positions to patch in one forward pass for a fixed layer.
PATCH_BATCH_SIZE = 16

# Patching granularity:
#  - "cell": one (layer, token) cell per intervention
#  - "token_all_layers": one token position patched at all selected layers
PATCH_SCOPE = "cell"

# Noise configuration for patched activations
NOISE_MODE = "gaussian"
NOISE_SCALE = 1.0
NOISE_SEED = 0
NOISE_SAMPLES_PER_CELL = 5

# Device for computation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

MODEL_PATH = str(resolve_model_path("Qwen2.5-32B"))
TOKENIZER_PATH = MODEL_PATH
SAVE_PLOTS = True


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
    parser = argparse.ArgumentParser(description="Run token/layer activation patching and export heatmaps + matrices.")
    parser.add_argument("--input-json", type=Path, default=INPUT_JSON, help="Input JSON with top-level 'pairs' or 'experiments'.")
    parser.add_argument("--output-root-dir", type=Path, default=OUTPUT_ROOT_DIR, help="Output root directory for run artifacts.")
    parser.add_argument("--output-summary-json", type=Path, default=None, help="Optional summary JSON path (default: <output-root-dir>/patching_summary.json).")

    parser.add_argument("--experiment-indices", type=str, default=None, help="Comma-separated experiment indices, e.g. '0,1,2'.")
    parser.add_argument("--experiment-ids", type=str, default=None, help="Comma-separated experiment IDs. Numeric values are also accepted and treated as experiment index and pair_id fallbacks.")
    parser.add_argument("--layers-to-patch", type=str, default=None, help="Comma-separated layer indices, or 'all'.")
    parser.add_argument("--layer-stride", type=int, default=LAYER_STRIDE, help="Patch every Nth layer after layer filtering (1 = all layers).")
    parser.add_argument("--token-positions-to-patch", type=str, default=None, help="Comma-separated token positions, or 'all'.")

    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Model path for AutoModelForCausalLM.")
    parser.add_argument("--tokenizer-path", type=str, default=TOKENIZER_PATH, help="Tokenizer path for AutoTokenizer.")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to run on (e.g., cuda, cpu).")
    parser.add_argument("--dtype", type=str, default="float16" if DTYPE == torch.float16 else "float32", choices=["float16", "float32"], help="Model dtype.")
    parser.add_argument("--noise-mode", type=str, default=NOISE_MODE, choices=["gaussian", "zero"], help="Noise replacement mode for patched activations.")
    parser.add_argument("--noise-scale", type=float, default=NOISE_SCALE, help="Scale factor applied to Gaussian noise.")
    parser.add_argument("--noise-seed", type=int, default=NOISE_SEED, help="Base seed for deterministic per-cell noise.")
    parser.add_argument("--noise-samples-per-cell", type=int, default=NOISE_SAMPLES_PER_CELL, help="Number of noisy patches to average per (layer, token) cell.")
    parser.add_argument("--patch-scope", type=str, default=PATCH_SCOPE, choices=["cell", "token_all_layers"], help="Patch one (layer, token) cell, or patch one token across all selected layers.")
    parser.add_argument("--patch-batch-size", type=int, default=PATCH_BATCH_SIZE, help="Number of token positions to patch per forward pass for a fixed layer.")
    parser.add_argument("--no-plots", action="store_true", help="Disable heatmap PNG generation and only write JSON outputs.")
    return parser


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

        # Skip pairs that post-processing flagged as tokenization/alignment failures.
        token_counts_equal = pair.get("post_process", {}).get("token_counts_equal")
        if token_counts_equal is False:
            continue

        # Use source + counterfactual traces to locate changed numeric spans.
        source_token_ids = source_block.get("tokens") if isinstance(source_block.get("tokens"), list) else []
        cf_token_ids = cf_block.get("tokens") if isinstance(cf_block.get("tokens"), list) else []
        source_text = source_block.get("generated_text", "")
        if not source_token_ids or not cf_token_ids:
            continue
        if len(source_token_ids) != len(cf_token_ids):
            continue

        # Find where the CoT starts ("Answer (step-by-step)")
        cot_marker = "Answer (step-by-step)"
        cot_start_pos = source_text.find(cot_marker)
        if cot_start_pos == -1:
            # If marker not found, no CoT values to process
            continue
        cot_start_pos += len(cot_marker)

        matched_values = (
            pair.get("post_process", {})
            .get("numeric_length_alignment", {})
            .get("matched_values", [])
        )

        # Filter to only values that appear in the CoT region (after "Answer (step-by-step)")
        cot_values = []
        if isinstance(matched_values, list):
            for mv in matched_values:
                if not isinstance(mv, dict):
                    continue
                # Check if value's source span starts in CoT region
                source_span = mv.get("source", {})
                cf_span = mv.get("counterfactual", {})
                span_start = source_span.get("span_start")
                if span_start is None:
                    span_start = source_span.get("span", {}).get("start")
                if span_start is not None and span_start >= cot_start_pos and _values_differ(source_span, cf_span):
                    cot_values.append(mv)

        # Sort in reverse order by span start position (final answer first)
        def _mv_span_start(mv: Dict) -> int:
            src = mv.get("source", {})
            val = src.get("span_start")
            if val is None:
                val = src.get("span", {}).get("start", 0)
            return int(val) if isinstance(val, int) else 0

        cot_values.sort(key=_mv_span_start, reverse=True)

        if cot_values:
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

                # Prefer scoring the first differing token inside the matched numeric span.
                if isinstance(src_tok_end, int) and isinstance(cf_tok_end, int):
                    span_start = max(0, min(src_tok_start, cf_tok_start))
                    span_end = min(len(source_token_ids), len(cf_token_ids), src_tok_end, cf_tok_end)
                else:
                    span_start = max(0, min(src_tok_start, cf_tok_start))
                    span_end = span_start + 1

                if span_end <= span_start:
                    continue

                trunc_idx = span_start
                found_diff = False
                for t in range(span_start, span_end):
                    if source_token_ids[t] != cf_token_ids[t]:
                        trunc_idx = t
                        found_diff = True
                        break

                if not found_diff:
                    continue

                if trunc_idx <= 0:
                    continue
                if trunc_idx >= len(source_token_ids):
                    continue
                if trunc_idx in used_trunc_indices:
                    continue

                used_trunc_indices.add(trunc_idx)

                experiments.append(
                    {
                        "experiment_id": f"pair{pair_id}_v{v_idx}",
                        "pair_id": pair_id,
                        "value_index": v_idx,
                        "experiment_type": "noise_value_match",
                        "truncation_token_index": trunc_idx,
                        "source": {
                            "token_ids": source_token_ids,
                            "score_token_id": int(source_token_ids[trunc_idx]),
                        },
                        "target": {
                            "token_ids": source_token_ids,
                            "score_token_id": int(source_token_ids[trunc_idx]),
                        },
                    }
                )

    return experiments


def build_experiments_from_traces(payload: Dict) -> List[Dict]:
    traces = payload.get("traces")
    if not isinstance(traces, list):
        return []

    experiments: List[Dict] = []
    for pidx, trace in enumerate(traces):
        if not isinstance(trace, dict):
            continue

        pair_id = trace.get("id", pidx)
        source_token_ids = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
        source_text = trace.get("generated_text", "")
        if not source_token_ids:
            continue

        cot_marker = "Answer (step-by-step)"
        cot_start_pos = source_text.find(cot_marker)
        if cot_start_pos == -1:
            continue
        cot_start_pos += len(cot_marker)

        trace_values = trace.get("values", [])
        cot_values = []
        if isinstance(trace_values, list):
            for value_obj in trace_values:
                if not isinstance(value_obj, dict):
                    continue
                span_start = value_obj.get("span_start")
                if isinstance(span_start, int) and span_start >= cot_start_pos:
                    cot_values.append(value_obj)

        def _value_span_start(value_obj: Dict) -> int:
            val = value_obj.get("span_start")
            return int(val) if isinstance(val, int) else 0

        cot_values.sort(key=_value_span_start, reverse=True)

        if cot_values:
            used_trunc_indices = set()
            for v_idx, value_obj in enumerate(cot_values):
                src_tok_start = value_obj.get("token_start")
                src_tok_end = value_obj.get("token_end")
                if not isinstance(src_tok_start, int):
                    continue

                if isinstance(src_tok_end, int):
                    trunc_idx = max(0, src_tok_start)
                else:
                    trunc_idx = max(0, src_tok_start)

                if trunc_idx <= 0 or trunc_idx >= len(source_token_ids):
                    continue
                if trunc_idx in used_trunc_indices:
                    continue

                used_trunc_indices.add(trunc_idx)

                experiments.append(
                    {
                        "experiment_id": f"trace{pair_id}_v{v_idx}",
                        "pair_id": pair_id,
                        "value_index": v_idx,
                        "experiment_type": "noise_value_match",
                        "truncation_token_index": trunc_idx,
                        "source": {
                            "token_ids": source_token_ids,
                            "score_token_id": int(source_token_ids[trunc_idx]),
                        },
                        "target": {
                            "token_ids": source_token_ids,
                            "score_token_id": int(source_token_ids[trunc_idx]),
                        },
                    }
                )
        else:
            usable_len = len(source_token_ids)
            if usable_len > 1:
                trunc_idx = usable_len - 1
                experiments.append(
                    {
                        "experiment_id": f"trace{pair_id}_tail",
                        "pair_id": pair_id,
                        "value_index": -1,
                        "experiment_type": "noise_tail_fallback",
                        "truncation_token_index": trunc_idx,
                        "source": {
                            "token_ids": source_token_ids,
                            "score_token_id": int(source_token_ids[trunc_idx]),
                        },
                        "target": {
                            "token_ids": source_token_ids,
                            "score_token_id": int(source_token_ids[trunc_idx]),
                        },
                    }
                )

    return experiments


def load_experiments_from_input(path: Path) -> List[Dict]:
    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and isinstance(payload.get("experiments"), list):
        return payload["experiments"]

    if isinstance(payload, dict) and isinstance(payload.get("traces"), list):
        return build_experiments_from_traces(payload)

    if isinstance(payload, dict) and isinstance(payload.get("pairs"), list):
        return build_experiments_from_pairs(payload)

    raise ValueError("Input JSON must contain top-level 'experiments', 'traces', or 'pairs'")


def stable_int_seed(*parts: object) -> int:
    joined = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)

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
    with torch.inference_mode():
        out = model(input_ids)
    return out.logits[0]


def forward_logits_batch(model, batch_token_ids: List[List[int]]) -> torch.Tensor:
    """Forward pass returning logits [batch, seq_len, vocab]."""
    input_ids = torch.tensor(batch_token_ids, device=DEVICE, dtype=torch.long)
    with torch.inference_mode():
        out = model(input_ids)
    return out.logits


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
    """Capture full hidden states for the clean trace at chosen layers."""
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


def make_noise_cell(reference_cell: torch.Tensor, experiment_id: str, layer_idx: int, token_pos: int) -> torch.Tensor:
    """Generate a single replacement cell for one layer/token position."""
    if NOISE_MODE == "zero":
        return torch.zeros_like(reference_cell)

    cell = reference_cell.float()
    cell_std = float(cell.std(unbiased=False).item())
    if not np.isfinite(cell_std) or cell_std <= 0:
        cell_std = 1.0

    seed = stable_int_seed(NOISE_SEED, experiment_id, layer_idx, token_pos)
    generator = torch.Generator(device=reference_cell.device)
    generator.manual_seed(seed)
    noise = torch.randn(reference_cell.shape, device=reference_cell.device, dtype=reference_cell.dtype, generator=generator)
    return noise * (cell_std * float(NOISE_SCALE))


def patched_logits_batch_for_layer(
    model,
    layers,
    target_token_ids: List[int],
    source_acts: Dict[int, torch.Tensor],
    layer_idx: int,
    experiment_id: str,
    token_positions: List[int],
    sample_idx: int = 0,
) -> torch.Tensor:
    """Forward the clean trace with many noisy patch cells on one layer.

    Each batch item corresponds to one token position and receives noise at
    only that single cell.
    """

    def _patch_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        src_hidden = source_acts[layer_idx].to(patched.device)
        batch_size = patched.shape[0]
        if batch_size != len(token_positions):
            raise ValueError(f"Batch size mismatch: got {batch_size}, expected {len(token_positions)}")

        for batch_idx, token_pos in enumerate(token_positions):
            if token_pos < patched.shape[1] and token_pos < src_hidden.shape[1]:
                reference_cell = src_hidden[:, token_pos, :]
                patched[batch_idx, token_pos, :] = make_noise_cell(reference_cell, f"{experiment_id}|sample{sample_idx}", layer_idx, token_pos)
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    hook = layers[layer_idx].register_forward_hook(_patch_hook)
    try:
        batch_token_ids = [target_token_ids for _ in token_positions]
        logits = forward_logits_batch(model, batch_token_ids)
    finally:
        hook.remove()
    return logits


def patched_logits_batch_for_token_all_layers(
    model,
    layers,
    target_token_ids: List[int],
    source_acts: Dict[int, torch.Tensor],
    layer_indices: List[int],
    experiment_id: str,
    token_positions: List[int],
    sample_idx: int = 0,
) -> torch.Tensor:
    """Forward clean trace with noisy patches at one token across all layers.

    Each batch item corresponds to one token position. For that item, the same
    token position is patched at every selected layer.
    """

    def make_patch_hook(layer_idx: int):
        def _patch_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            patched = hidden.clone()
            src_hidden = source_acts[layer_idx].to(patched.device)
            batch_size = patched.shape[0]
            if batch_size != len(token_positions):
                raise ValueError(f"Batch size mismatch: got {batch_size}, expected {len(token_positions)}")

            for batch_idx, token_pos in enumerate(token_positions):
                if token_pos < patched.shape[1] and token_pos < src_hidden.shape[1]:
                    reference_cell = src_hidden[:, token_pos, :]
                    patched[batch_idx, token_pos, :] = make_noise_cell(
                        reference_cell,
                        f"{experiment_id}|sample{sample_idx}",
                        layer_idx,
                        token_pos,
                    )
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched

        return _patch_hook

    hooks = []
    try:
        for layer_idx in layer_indices:
            hooks.append(layers[layer_idx].register_forward_hook(make_patch_hook(layer_idx)))
        batch_token_ids = [target_token_ids for _ in token_positions]
        logits = forward_logits_batch(model, batch_token_ids)
    finally:
        for hook in hooks:
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
    y_tick_labels = ["all" if int(layer_indices[i]) == -1 else str(layer_indices[i]) for i in y_ticks]
    ax.set_yticklabels(y_tick_labels, fontsize=8)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def build_heatmap_title(exp_id: str, pair_id: int, exp_type: str, trunc_idx: int, scored_token_index: int) -> str:
    return (
        f"{exp_id} | pair {pair_id} | type {exp_type} | trunc {trunc_idx}\n"
        f"NOISE delta: log P(true token@{scored_token_index})"
    )


def main():
    global INPUT_JSON
    global OUTPUT_ROOT_DIR
    global OUTPUT_SUMMARY_JSON
    global EXPERIMENT_INDICES
    global EXPERIMENT_IDS
    global LAYERS_TO_PATCH
    global LAYER_STRIDE
    global TOKEN_POSITIONS_TO_PATCH
    global MODEL_PATH
    global TOKENIZER_PATH
    global DEVICE
    global DTYPE
    global SAVE_PLOTS
    global NOISE_MODE
    global NOISE_SCALE
    global NOISE_SEED
    global NOISE_SAMPLES_PER_CELL
    global PATCH_BATCH_SIZE
    global PATCH_SCOPE

    args = build_arg_parser().parse_args()
    INPUT_JSON = args.input_json
    OUTPUT_ROOT_DIR = args.output_root_dir
    OUTPUT_SUMMARY_JSON = args.output_summary_json or (OUTPUT_ROOT_DIR / "patching_summary.json")

    EXPERIMENT_INDICES = parse_int_list(args.experiment_indices)
    EXPERIMENT_IDS = parse_str_list(args.experiment_ids)
    LAYERS_TO_PATCH = parse_int_list(args.layers_to_patch)
    LAYER_STRIDE = max(1, int(args.layer_stride))
    TOKEN_POSITIONS_TO_PATCH = parse_int_list(args.token_positions_to_patch)

    MODEL_PATH = args.model_path
    TOKENIZER_PATH = args.tokenizer_path
    DEVICE = args.device
    DTYPE = torch.float16 if args.dtype == "float16" else torch.float32
    SAVE_PLOTS = not args.no_plots
    NOISE_MODE = args.noise_mode
    NOISE_SCALE = args.noise_scale
    NOISE_SEED = args.noise_seed
    NOISE_SAMPLES_PER_CELL = args.noise_samples_per_cell
    PATCH_SCOPE = args.patch_scope
    PATCH_BATCH_SIZE = args.patch_batch_size

    print("\n" + "=" * 100)
    print("ACTIVATION PATCHING ENGINE")
    print("=" * 100)

    print(f"\nConfiguration:")
    print(f"  Input JSON: {INPUT_JSON}")
    print(f"  Output root dir: {OUTPUT_ROOT_DIR}")
    print(f"  Summary JSON: {OUTPUT_SUMMARY_JSON}")
    print(f"  Experiment IDs filter: {EXPERIMENT_IDS if EXPERIMENT_IDS else 'none'}")
    print(f"  Layers to patch: {LAYERS_TO_PATCH if LAYERS_TO_PATCH else 'all'}")
    print(f"  Layer stride: every {LAYER_STRIDE} layer(s)")
    print(f"  Token positions to patch: {TOKEN_POSITIONS_TO_PATCH if TOKEN_POSITIONS_TO_PATCH else 'all'}")
    print(f"  Patch scope: {PATCH_SCOPE}")
    print(f"  Noise mode: {NOISE_MODE} | scale: {NOISE_SCALE} | seed: {NOISE_SEED}")
    print(f"  Noise samples/cell: {NOISE_SAMPLES_PER_CELL}")
    print(f"  Patch batch size: {PATCH_BATCH_SIZE}")
    print(f"  Device: {DEVICE}, Dtype: {DTYPE}")
    print(f"  Save plots: {SAVE_PLOTS}")

    model, tokenizer = load_model_and_tokenizer()
    _ = tokenizer
    layers = get_transformer_layers(model)
    n_layers_total = len(layers)
    print(f"  Resolved transformer layers: {n_layers_total}")

    print(f"\nLoading experiments from {INPUT_JSON}...")
    experiments = load_experiments_from_input(INPUT_JSON)
    print(f"Loaded {len(experiments)} experiments")

    if EXPERIMENT_IDS is not None:
        id_to_index = {
            exp.get("experiment_id", f"idx_{i}"): i for i, exp in enumerate(experiments)
        }
        exp_indices = []
        for exp_id in EXPERIMENT_IDS:
            if exp_id in id_to_index:
                exp_indices.append(id_to_index[exp_id])
                continue

            matched = False
            try:
                numeric_id = int(exp_id)
            except ValueError:
                numeric_id = None

            if numeric_id is not None:
                if 0 <= numeric_id < len(experiments):
                    exp_indices.append(numeric_id)
                    matched = True
                pair_matches = [
                    i for i, ex in enumerate(experiments)
                    if ex.get("pair_id") == numeric_id
                ]
                if pair_matches:
                    exp_indices.extend(pair_matches)
                    matched = True

            if not matched:
                print(f"  Warning: experiment selector '{exp_id}' not found. Skipping.")

        # Preserve insertion order while removing duplicates.
        exp_indices = list(dict.fromkeys(exp_indices))
    elif EXPERIMENT_INDICES is None:
        exp_indices = list(range(len(experiments)))
    else:
        exp_indices = EXPERIMENT_INDICES

    print(f"Running {len(exp_indices)} experiments...")

    run_records = []
    pair_merged_payloads = {}
    OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    for exp_idx in exp_indices:
        if exp_idx >= len(experiments):
            print(f"  Warning: experiment index {exp_idx} out of range. Skipping.")
            continue

        exp = experiments[exp_idx]
        exp_id = exp.get("experiment_id", f"idx_{exp_idx}")
        pair_id = exp["pair_id"]
        value_index = exp.get("value_index", -1)
        exp_type = exp["experiment_type"]
        trunc_idx = exp["truncation_token_index"]

        print(f"\n  [{exp_idx}] {exp_id} | Pair {pair_id}, Type {exp_type}, Truncation {trunc_idx}")

        source_token_ids = exp["source"]["token_ids"]
        base_token_ids = source_token_ids

        usable_len = len(source_token_ids)
        if usable_len < 1:
            print("    Warning: usable token length < 1, skipping.")
            continue

        source_token_ids = source_token_ids[:usable_len]
        base_token_ids = base_token_ids[:usable_len]

        layer_indices = list(range(n_layers_total)) if LAYERS_TO_PATCH is None else list(LAYERS_TO_PATCH)
        layer_indices = [l for l in layer_indices if 0 <= l < n_layers_total]
        if LAYER_STRIDE > 1:
            layer_indices = [l for l in layer_indices if (l % LAYER_STRIDE) == 0]
        if not layer_indices:
            print("    Warning: no valid layers to patch, skipping.")
            continue

        # Limit token positions to patch up to truncation index (don't patch past truncation point)
        if TOKEN_POSITIONS_TO_PATCH is None:
            pos_indices = list(range(trunc_idx))
        else:
            pos_indices = [p for p in TOKEN_POSITIONS_TO_PATCH if 0 <= p < trunc_idx]
        
        if not pos_indices:
            print(f"    Warning: no valid token positions to patch (truncation at {trunc_idx}), skipping.")
            continue
        source_score_token_id = exp["source"].get("score_token_id")
        if source_score_token_id is None:
            print("    Warning: missing score_token_id in experiment data. Ensure INPUT_JSON includes score targets or aligned pairs with matched token spans. Skipping.")
            continue

        print(f"    SOURCE/TARGET usable length: {usable_len}")
        print(f"    Layers patched: {len(layer_indices)} | Positions patched: {len(pos_indices)}")

        print("    Phase 1: capture SOURCE activations (all selected layers)")
        source_acts = capture_source_activations(model, layers, source_token_ids, layer_indices)

        print("    Phase 2: baseline clean forward and true-next-token logprob")
        baseline_logits = forward_logits(model, base_token_ids)
        baseline_true_scores = compute_next_token_logprobs(baseline_logits, source_token_ids, usable_len)
        # Truncation index n is exclusive context [0..n-1]; score token index n
        # from logits at position n-1.
        if not isinstance(trunc_idx, int) or trunc_idx <= 0 or trunc_idx >= usable_len:
            print(f"    Warning: invalid truncation index {trunc_idx} for usable_len={usable_len}, skipping.")
            continue
        score_pos = trunc_idx - 1
        scored_token_index = trunc_idx
        baseline_true_score = score_logprob_at_position(baseline_logits, score_pos, int(source_score_token_id))

        if PATCH_SCOPE == "cell":
            matrix_layer_indices = list(layer_indices)
            noise_delta_matrix = np.full((len(matrix_layer_indices), len(pos_indices)), np.nan, dtype=np.float32)
            noise_delta_std_matrix = np.full((len(matrix_layer_indices), len(pos_indices)), np.nan, dtype=np.float32)

            print("    Phase 3: batched multi-sample noisy patching over (layer, position)")
            for li, layer_idx in enumerate(matrix_layer_indices):
                for batch_start in range(0, len(pos_indices), PATCH_BATCH_SIZE):
                    batch_positions = pos_indices[batch_start : batch_start + PATCH_BATCH_SIZE]

                    trial_deltas_by_pos = {pos: [] for pos in batch_positions}
                    for sample_idx in range(NOISE_SAMPLES_PER_CELL):
                        patched_logits = patched_logits_batch_for_layer(
                            model=model,
                            layers=layers,
                            target_token_ids=base_token_ids,
                            source_acts=source_acts,
                            layer_idx=layer_idx,
                            experiment_id=exp_id,
                            token_positions=batch_positions,
                            sample_idx=sample_idx,
                        )

                        for batch_offset, pos in enumerate(batch_positions):
                            patched_true_score = score_logprob_at_position(patched_logits[batch_offset], score_pos, int(source_score_token_id))
                            trial_deltas_by_pos[pos].append(patched_true_score - baseline_true_score)

                    for batch_offset, pos in enumerate(batch_positions):
                        pi = batch_start + batch_offset
                        trial_array = np.asarray(trial_deltas_by_pos[pos], dtype=np.float32)
                        noise_delta_matrix[li, pi] = float(np.mean(trial_array))
                        noise_delta_std_matrix[li, pi] = float(np.std(trial_array, ddof=0))
        elif PATCH_SCOPE == "token_all_layers":
            # Aggregate over all selected layers in one intervention per token.
            matrix_layer_indices = [-1]
            noise_delta_matrix = np.full((1, len(pos_indices)), np.nan, dtype=np.float32)
            noise_delta_std_matrix = np.full((1, len(pos_indices)), np.nan, dtype=np.float32)

            print("    Phase 3: batched multi-sample noisy patching over (position), patching all selected layers")
            for batch_start in range(0, len(pos_indices), PATCH_BATCH_SIZE):
                batch_positions = pos_indices[batch_start : batch_start + PATCH_BATCH_SIZE]

                trial_deltas_by_pos = {pos: [] for pos in batch_positions}
                for sample_idx in range(NOISE_SAMPLES_PER_CELL):
                    patched_logits = patched_logits_batch_for_token_all_layers(
                        model=model,
                        layers=layers,
                        target_token_ids=base_token_ids,
                        source_acts=source_acts,
                        layer_indices=layer_indices,
                        experiment_id=exp_id,
                        token_positions=batch_positions,
                        sample_idx=sample_idx,
                    )

                    for batch_offset, pos in enumerate(batch_positions):
                        patched_true_score = score_logprob_at_position(patched_logits[batch_offset], score_pos, int(source_score_token_id))
                        trial_deltas_by_pos[pos].append(patched_true_score - baseline_true_score)

                for batch_offset, pos in enumerate(batch_positions):
                    pi = batch_start + batch_offset
                    trial_array = np.asarray(trial_deltas_by_pos[pos], dtype=np.float32)
                    noise_delta_matrix[0, pi] = float(np.mean(trial_array))
                    noise_delta_std_matrix[0, pi] = float(np.std(trial_array, ddof=0))
        else:
            raise ValueError(f"Unknown patch scope: {PATCH_SCOPE}")

        # All outputs for a pair go into a single pair{id} directory
        exp_dir = OUTPUT_ROOT_DIR / f"pair{pair_id}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Include value_index in filenames to distinguish different numeric values within the pair
        value_suffix = f"_v{value_index}" if value_index >= 0 else "_tail"
        noise_heatmap_png = exp_dir / f"noise_heatmap{value_suffix}_t{trunc_idx}.png"
        pair_json = exp_dir / "pair_matrices.json"
        x_tick_labels = build_x_tick_labels(tokenizer, source_token_ids, pos_indices)

        finite_vals = noise_delta_matrix[np.isfinite(noise_delta_matrix)]
        noise_vmax = float(np.abs(finite_vals).max()) if finite_vals.size > 0 else 1.0
        if noise_vmax <= 0:
            noise_vmax = 1.0

        noise_title = build_heatmap_title(exp_id, pair_id, exp_type, trunc_idx, scored_token_index)

        if SAVE_PLOTS:
            plot_single_heatmap(
                matrix=noise_delta_matrix,
                title=noise_title,
                out_png=noise_heatmap_png,
                x_labels=x_tick_labels,
                layer_indices=matrix_layer_indices,
                cbar_label="Delta log P(true next token)",
                vmax=noise_vmax,
            )

        matrix_payload = {
            "experiment_id": exp_id,
            "pair_id": pair_id,
            "value_index": value_index,
            "experiment_type": exp_type,
            "truncation_token_index": trunc_idx,
            "usable_len": usable_len,
            "layer_indices": matrix_layer_indices,
            "patched_layer_indices": layer_indices,
            "position_indices": pos_indices,
            "score_target_definition": "token at truncation index n (predicted from position n-1)",
            "score_position_used": score_pos,
            "scored_token_index": scored_token_index,
            "scored_source_token_id": int(source_score_token_id),
            "scored_source_token_text": tokenizer.decode([int(source_score_token_id)]),
            "patch_scope": PATCH_SCOPE,
            "noise_mode": NOISE_MODE,
            "noise_scale": NOISE_SCALE,
            "noise_seed": NOISE_SEED,
            "noise_samples_per_cell": NOISE_SAMPLES_PER_CELL,
            "patch_batch_size": PATCH_BATCH_SIZE,
            "baseline_true_logprob": baseline_true_score,
            "x_tick_labels": x_tick_labels,
            "baseline_next_true_logprobs": baseline_true_scores,
            "noise_delta_matrix": noise_delta_matrix.tolist(),
            "noise_delta_std_matrix": noise_delta_std_matrix.tolist(),
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
        pair_merged_payloads[pair_key]["entries"].append(matrix_payload)

        # Flush per-pair JSON incrementally so partial/interrupted runs still
        # leave usable pair-level data artifacts on disk.
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
                "noise_heatmap_png": str(noise_heatmap_png) if SAVE_PLOTS else None,
                "pair_json": str(pair_json),
                "usable_len": usable_len,
                "n_layers": len(layer_indices),
                "n_matrix_rows": int(noise_delta_matrix.shape[0]),
                "n_positions": len(pos_indices),
                "patch_scope": PATCH_SCOPE,
            }
        )
        if SAVE_PLOTS:
            print(f"    Saved: {noise_heatmap_png}")

    for pair_key, payload in pair_merged_payloads.items():
        pair_dir = OUTPUT_ROOT_DIR / pair_key
        pair_json = pair_dir / "pair_matrices.json"
        entries = payload.get("entries", [])
        entries.sort(key=lambda x: x.get("truncation_token_index", 0), reverse=True)
        payload["n_entries"] = len(entries)
        with open(pair_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved merged pair JSON: {pair_json}")

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
            "layer_stride": LAYER_STRIDE,
            "positions_to_patch": TOKEN_POSITIONS_TO_PATCH if TOKEN_POSITIONS_TO_PATCH else "all",
            "score_target_definition": "token at truncation index n (predicted from position n-1)",
            "patch_scope": PATCH_SCOPE,
            "noise_mode": NOISE_MODE,
            "noise_scale": NOISE_SCALE,
            "noise_seed": NOISE_SEED,
            "noise_samples_per_cell": NOISE_SAMPLES_PER_CELL,
            "patch_batch_size": PATCH_BATCH_SIZE,
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
