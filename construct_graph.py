#!/usr/bin/env python3
"""Construct value-level causal graphs from activation patching matrices.

This script reads matrix JSON files produced by intervene_graph.py and builds a
value graph per pair:
- Nodes are values (v0, v1, ...), where v0 is expected to be the final answer.
- Directed edges parent -> child are inferred from token-level causal scores.
- Traversal starts at v0 and iteratively walks backward to parents.

Selection methods:
- topk: fixed number of strongest token positions
- quantile: keep token positions above a score quantile
- fdr: robust z-score + Benjamini-Hochberg FDR control (adaptive parent count)
"""

import argparse
import json
import math
import re
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


MATRIX_FILE_RE = re.compile(r"^matrix_(v(?P<vidx>\d+)|tail)_t(?P<trunc>\d+)\.json$")


def load_aligned_pairs(aligned_pairs_path: Path) -> Dict[str, Dict]:
    """Load aligned pairs from JSON file, indexed by pair_id."""
    if not aligned_pairs_path.exists():
        return {}
    
    try:
        with open(aligned_pairs_path, "r") as f:
            payload = json.load(f)
        
        pairs_dict = {}
        pairs_list = payload.get("pairs", []) if isinstance(payload, dict) else payload
        
        for pair in pairs_list:
            if not isinstance(pair, dict):
                continue
            pair_id = str(pair.get("pair_id", pair.get("id", "")))
            if pair_id:
                pairs_dict[pair_id] = pair
        
        return pairs_dict
    except Exception as e:
        print(f"Warning: Could not load aligned_pairs.json: {e}")
        return {}


def get_pair_trace(pair_id: str, aligned_pairs_dict: Dict[str, Dict]) -> Optional[str]:
    """Extract the base trace text from a pair record."""
    pair = aligned_pairs_dict.get(pair_id)
    if not pair:
        return None
    
    # Try various fields where the source text might be stored.
    nested_pair = pair.get("pair", {}) if isinstance(pair.get("pair"), dict) else {}
    nested_source = nested_pair.get("source", {}) if isinstance(nested_pair.get("source"), dict) else {}

    trace_text = (
        nested_source.get("generated_text") or
        nested_source.get("text") or
        pair.get("source_text") or
        pair.get("source") or
        pair.get("base_text") or
        pair.get("prompt") or
        ""
    )
    
    return trace_text if trace_text else None


def get_pair_record(pair_id: str, aligned_pairs_dict: Dict[str, Dict]) -> Optional[Dict]:
    pair = aligned_pairs_dict.get(pair_id)
    if isinstance(pair, dict):
        return pair
    return None


def build_value_label_by_index(pair_record: Optional[Dict]) -> Dict[int, str]:
    """Map graph value index (v0, v1, ...) to value_text from source trace values.

    Value indices in patching are assigned from the end of the value list, so v0 is
    the last detected numeric value in the source trace.
    """
    if not isinstance(pair_record, dict):
        return {}

    nested_pair = pair_record.get("pair", {}) if isinstance(pair_record.get("pair"), dict) else {}
    nested_source = nested_pair.get("source", {}) if isinstance(nested_pair.get("source"), dict) else {}
    values = nested_source.get("values", []) if isinstance(nested_source.get("values"), list) else []

    out: Dict[int, str] = {}
    for vidx, value_obj in enumerate(reversed(values)):
        if not isinstance(value_obj, dict):
            continue
        value_text = str(value_obj.get("value_text", "")).strip()
        if value_text:
            out[vidx] = value_text
    return out


def build_value_token_ranges_by_index(pair_record: Optional[Dict]) -> Dict[int, Tuple[int, int]]:
    """Map graph value index (v0, v1, ...) to source token span [start, end)."""
    if not isinstance(pair_record, dict):
        return {}

    nested_pair = pair_record.get("pair", {}) if isinstance(pair_record.get("pair"), dict) else {}
    nested_source = nested_pair.get("source", {}) if isinstance(nested_pair.get("source"), dict) else {}
    values = nested_source.get("values", []) if isinstance(nested_source.get("values"), list) else []

    out: Dict[int, Tuple[int, int]] = {}
    for vidx, value_obj in enumerate(reversed(values)):
        if not isinstance(value_obj, dict):
            continue
        ts = value_obj.get("token_start")
        te = value_obj.get("token_end")
        if isinstance(ts, int) and isinstance(te, int) and te > ts:
            out[vidx] = (ts, te)
    return out


def build_value_alignment_line(pair_record: Optional[Dict], trace_text: str) -> Optional[str]:
    """Build a monospaced line with vN labels aligned under the original value spans."""
    if not isinstance(pair_record, dict):
        return None

    nested_pair = pair_record.get("pair", {}) if isinstance(pair_record.get("pair"), dict) else {}
    nested_source = nested_pair.get("source", {}) if isinstance(nested_pair.get("source"), dict) else {}
    text = str(trace_text or nested_source.get("generated_text", ""))
    values = nested_source.get("values", []) if isinstance(nested_source.get("values"), list) else []

    if not text or not values:
        return None

    line = [" "] * len(text)
    rev_values = list(reversed(values))
    for vidx, value_obj in enumerate(rev_values):
        if not isinstance(value_obj, dict):
            continue
        span_start = value_obj.get("span_start")
        span_end = value_obj.get("span_end")
        label = f"v{vidx}"
        if not isinstance(span_start, int) or not isinstance(span_end, int):
            continue

        # Exact placement: write the label starting at the original span start.
        if span_start < 0 or span_start >= len(text):
            continue
        for i, ch in enumerate(label):
            pos = span_start + i
            if pos < len(line):
                line[pos] = ch

    return "".join(line).rstrip()


def build_excluded_shared_value_indices(pair_record: Optional[Dict]) -> Set[int]:
    """Identify values shared across source/counterfactual that only appear in the CoT.

    Shared prompt values are kept. Shared numeric literals that begin after the prompt
    are usually step numbers or formula constants, and should not become graph nodes.
    """
    if not isinstance(pair_record, dict):
        return set()

    nested_pair = pair_record.get("pair", {}) if isinstance(pair_record.get("pair"), dict) else {}
    nested_source = nested_pair.get("source", {}) if isinstance(nested_pair.get("source"), dict) else {}
    nested_cf = nested_pair.get("counterfactual", {}) if isinstance(nested_pair.get("counterfactual"), dict) else {}

    source_values = nested_source.get("values", []) if isinstance(nested_source.get("values"), list) else []
    cf_values = nested_cf.get("values", []) if isinstance(nested_cf.get("values"), list) else []
    prompt_text = str(pair_record.get("prompt", "") or "")
    prompt_len = len(prompt_text)

    excluded: Set[int] = set()
    for vidx, (src, cf) in enumerate(zip(source_values, cf_values)):
        if not isinstance(src, dict) or not isinstance(cf, dict):
            continue
        src_norm = str(src.get("normalized_value", "")).strip()
        cf_norm = str(cf.get("normalized_value", "")).strip()
        if not src_norm or src_norm != cf_norm:
            continue
        span_start = src.get("span_start")
        if isinstance(span_start, int) and span_start >= prompt_len:
            excluded.add(vidx)

    return excluded


def build_excluded_shared_value_metadata(pair_record: Optional[Dict]) -> List[Dict[str, object]]:
    """Return structured metadata for shared CoT-only values that are excluded from the graph."""
    if not isinstance(pair_record, dict):
        return []

    nested_pair = pair_record.get("pair", {}) if isinstance(pair_record.get("pair"), dict) else {}
    nested_source = nested_pair.get("source", {}) if isinstance(nested_pair.get("source"), dict) else {}
    nested_cf = nested_pair.get("counterfactual", {}) if isinstance(nested_pair.get("counterfactual"), dict) else {}

    source_values = nested_source.get("values", []) if isinstance(nested_source.get("values"), list) else []
    cf_values = nested_cf.get("values", []) if isinstance(nested_cf.get("values"), list) else []
    prompt_text = str(pair_record.get("prompt", "") or "")
    prompt_len = len(prompt_text)

    excluded: List[Dict[str, object]] = []
    for vidx, (src, cf) in enumerate(zip(source_values, cf_values)):
        if not isinstance(src, dict) or not isinstance(cf, dict):
            continue

        src_norm = str(src.get("normalized_value", "")).strip()
        cf_norm = str(cf.get("normalized_value", "")).strip()
        if not src_norm or src_norm != cf_norm:
            continue

        span_start = src.get("span_start")
        if not isinstance(span_start, int) or span_start < prompt_len:
            continue

        excluded.append(
            {
                "value_index": vidx,
                "value_text": src.get("value_text"),
                "normalized_value": src_norm,
                "span_start": span_start,
                "span_end": src.get("span_end"),
                "reason": "shared_cot_constant",
            }
        )

    return excluded


def build_truncation_label_row(graph: Dict, width: int = 120) -> str:
    """Build a monospaced row of value labels positioned by truncation index.

    Kept for graph-level summaries, but footer rendering now prefers value-span alignment.
    """
    nodes = graph.get("nodes", [])
    if not nodes or width <= 0:
        return ""

    ordered_nodes = sorted(
        nodes,
        key=lambda n: (int(n.get("truncation_token_index", 0)), str(n.get("id", ""))),
    )
    max_trunc = max(int(n.get("truncation_token_index", 0)) for n in ordered_nodes)
    max_trunc = max(1, max_trunc)

    canvas = [" "] * width
    occupied = [False] * width

    for node in ordered_nodes:
        trunc = int(node.get("truncation_token_index", 0))
        value_indices = [int(v) for v in node.get("value_indices", [])]
        if not value_indices:
            continue

        label = ",".join([f"v{vidx}" for vidx in value_indices])
        if not label:
            continue

        anchor = int(round((trunc / max_trunc) * (width - 1)))
        start = max(0, min(width - len(label), anchor - len(label) // 2))

        # Shift right slightly if this label would collide with an earlier one.
        placed = False
        for offset in range(width):
            candidate = min(width - len(label), start + offset)
            if not any(occupied[candidate : candidate + len(label)]):
                for i, ch in enumerate(label):
                    canvas[candidate + i] = ch
                    occupied[candidate + i] = True
                placed = True
                break
        if not placed:
            for i, ch in enumerate(label[:width]):
                canvas[i] = ch

    return "".join(canvas).rstrip()


@dataclass
class MatrixRecord:
    pair_id: str
    node_id: str
    value_index: int
    truncation_token_index: int
    position_indices: List[int]
    diff_delta_matrix: np.ndarray


@dataclass
class GraphBuildConfig:
    layer_agg: str
    selection_method: str
    top_k: int
    quantile: float
    fdr_q: float
    min_tokens: int
    relative_edge_threshold: float
    max_depth: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build value graphs from patching matrix JSON files.")
    p.add_argument("--patch-runs-dir", type=Path, required=True, help="Directory containing pair*/matrix_*.json outputs.")
    p.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: <patch-runs-dir>/graphs).")
    p.add_argument("--pair-id", type=str, default=None, help="Only process one pair id, e.g. 0 or pair0.")

    p.add_argument("--layer-agg", choices=["mean_abs", "max_abs", "mean_signed"], default="mean_abs")
    p.add_argument("--selection-method", choices=["topk", "quantile", "fdr"], default="fdr")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--quantile", type=float, default=0.9)
    p.add_argument("--fdr-q", type=float, default=0.5)
    p.add_argument("--min-tokens", type=int, default=1, help="Minimum number of selected token positions per child node.")
    p.add_argument(
        "--relative-edge-threshold",
        type=float,
        default=0.2,
        help="Keep parent edges with weight >= threshold * max_parent_weight for each child.",
    )
    p.add_argument("--max-depth", type=int, default=100, help="Safety cap for BFS depth.")
    p.add_argument(
        "--render",
        choices=["none", "png", "svg"],
        default="png",
        help="Render graph image from DOT (default: png). Uses Graphviz if available.",
    )
    p.add_argument(
        "--layout",
        choices=["dot", "neato", "fdp", "sfdp", "twopi", "circo"],
        default="dot",
        help="Graphviz layout engine when rendering with Graphviz.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Image dpi for matplotlib fallback renderer.",
    )
    return p.parse_args()


def aggregate_token_scores(diff_delta_matrix: np.ndarray, layer_agg: str) -> np.ndarray:
    if diff_delta_matrix.ndim != 2:
        raise ValueError("diff_delta_matrix must be rank-2 [layers, positions]")

    if layer_agg == "mean_abs":
        return np.mean(np.abs(diff_delta_matrix), axis=0)
    if layer_agg == "max_abs":
        return np.max(np.abs(diff_delta_matrix), axis=0)
    if layer_agg == "mean_signed":
        return np.mean(diff_delta_matrix, axis=0)
    raise ValueError(f"Unknown layer aggregation: {layer_agg}")


def robust_zscores(x: np.ndarray) -> np.ndarray:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad <= 1e-12:
        std = float(np.std(x))
        if std <= 1e-12:
            return np.zeros_like(x)
        return (x - med) / std
    return (x - med) / (1.4826 * mad)


def benjamini_hochberg(pvals: np.ndarray, q: float) -> np.ndarray:
    n = pvals.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = q * (np.arange(1, n + 1) / n)
    passed = ranked <= thresholds
    if not np.any(passed):
        return np.zeros(n, dtype=bool)
    k = int(np.max(np.where(passed)[0]))
    cutoff = ranked[k]
    return pvals <= cutoff


def select_token_positions(scores: np.ndarray, method: str, top_k: int, quantile: float, fdr_q: float, min_tokens: int) -> np.ndarray:
    n = scores.size
    if n == 0:
        return np.array([], dtype=int)

    magnitudes = np.abs(scores)

    if method == "topk":
        k = max(1, min(top_k, n))
        idx = np.argsort(-magnitudes)[:k]
        return np.sort(idx)

    if method == "quantile":
        qv = float(np.quantile(magnitudes, quantile))
        idx = np.where(magnitudes >= qv)[0]
        if idx.size < min_tokens:
            k = min(min_tokens, n)
            idx = np.argsort(-magnitudes)[:k]
        return np.sort(idx)

    if method == "fdr":
        z = robust_zscores(magnitudes)
        # Two-sided normal p-value using erfc.
        pvals = np.array([math.erfc(abs(float(v)) / math.sqrt(2.0)) for v in z], dtype=np.float64)
        keep = benjamini_hochberg(pvals, fdr_q)
        idx = np.where(keep)[0]
        if idx.size < min_tokens:
            k = min(min_tokens, n)
            idx = np.argsort(-magnitudes)[:k]
        return np.sort(idx)

    raise ValueError(f"Unknown selection method: {method}")


def parse_matrix_file(path: Path, pair_id: str) -> Optional[MatrixRecord]:
    m = MATRIX_FILE_RE.match(path.name)
    if not m:
        return None

    with open(path, "r") as f:
        payload = json.load(f)

    raw_vidx = m.group("vidx")
    value_index = int(raw_vidx) if raw_vidx is not None else -1
    node_id = f"v{value_index}" if value_index >= 0 else "tail"

    diff = np.array(payload.get("diff_delta_matrix", []), dtype=np.float32)
    pos = payload.get("position_indices", [])
    trunc = int(payload.get("truncation_token_index", int(m.group("trunc"))))

    if diff.size == 0 or not isinstance(pos, list):
        return None

    return MatrixRecord(
        pair_id=pair_id,
        node_id=node_id,
        value_index=value_index,
        truncation_token_index=trunc,
        position_indices=[int(p) for p in pos],
        diff_delta_matrix=diff,
    )


def parse_matrix_payload(payload: Dict, pair_id: str) -> Optional[MatrixRecord]:
    value_index = payload.get("value_index")
    if not isinstance(value_index, int):
        exp_id = str(payload.get("experiment_id", ""))
        m = re.search(r"_v(\d+)$", exp_id)
        value_index = int(m.group(1)) if m else -1

    node_id = f"v{value_index}" if value_index >= 0 else "tail"
    diff = np.array(payload.get("diff_delta_matrix", []), dtype=np.float32)
    pos = payload.get("position_indices", [])
    trunc = payload.get("truncation_token_index")

    if diff.size == 0 or not isinstance(pos, list) or not isinstance(trunc, int):
        return None

    return MatrixRecord(
        pair_id=pair_id,
        node_id=node_id,
        value_index=value_index,
        truncation_token_index=trunc,
        position_indices=[int(p) for p in pos],
        diff_delta_matrix=diff,
    )


def read_pair_matrices(pair_dir: Path) -> Dict[str, MatrixRecord]:
    pair_id = pair_dir.name.replace("pair", "")
    out: Dict[str, MatrixRecord] = {}

    merged_file = pair_dir / "pair_matrices.json"
    if merged_file.exists():
        with open(merged_file, "r") as f:
            merged_payload = json.load(f)
        entries = merged_payload.get("entries", [])
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                rec = parse_matrix_payload(entry, pair_id=pair_id)
                if rec is None:
                    continue
                out[rec.node_id] = rec

    if not out:
        # Backward compatibility with legacy per-truncation JSON files.
        for fp in sorted(pair_dir.glob("matrix_*.json")):
            rec = parse_matrix_file(fp, pair_id=pair_id)
            if rec is None:
                continue
            out[rec.node_id] = rec

    return out


def map_tokens_to_parent_nodes(
    token_positions: List[int],
    child_node_id: str,
    node_to_trunc: Dict[str, int],
) -> List[str]:
    # Parent candidates must occur earlier in sequence than child.
    child_trunc = node_to_trunc[child_node_id]
    parent_nodes = [nid for nid, t in node_to_trunc.items() if t < child_trunc and nid != child_node_id]
    if not parent_nodes:
        return []

    # Map each token position to nearest parent value anchor not after token.
    mapped: List[str] = []
    for tp in token_positions:
        feasible = [(nid, node_to_trunc[nid]) for nid in parent_nodes if node_to_trunc[nid] <= tp]
        if not feasible:
            continue
        chosen = max(feasible, key=lambda x: x[1])[0]
        mapped.append(chosen)
    return mapped


def _validate_records_by_value(records: Dict[str, MatrixRecord]) -> Dict[str, MatrixRecord]:
    """Ensure each value_index appears at most once. If duplicates exist, keep the first."""
    by_value: Dict[int, MatrixRecord] = {}
    for rec in records.values():
        if rec.value_index < 0:
            by_value[rec.value_index] = rec
        elif rec.value_index not in by_value:
            by_value[rec.value_index] = rec
    
    return {rec.node_id: rec for rec in by_value.values()}


def build_graph_for_pair(
    records: Dict[str, MatrixRecord],
    cfg: GraphBuildConfig,
    value_label_by_index: Optional[Dict[int, str]] = None,
    value_token_ranges_by_index: Optional[Dict[int, Tuple[int, int]]] = None,
    excluded_value_indices: Optional[Set[int]] = None,
    excluded_shared_value_metadata: Optional[List[Dict[str, object]]] = None,
) -> Dict:
    if value_label_by_index is None:
        value_label_by_index = {}
    if value_token_ranges_by_index is None:
        value_token_ranges_by_index = {}
    if excluded_value_indices is None:
        excluded_value_indices = set()
    if excluded_shared_value_metadata is None:
        excluded_shared_value_metadata = []

    records = {
        nid: rec
        for nid, rec in records.items()
        if rec.value_index not in excluded_value_indices
    }

    records = _validate_records_by_value(records)

    # Each record is its own node; no collapsing by truncation index
    node_to_value_indices: Dict[str, List[int]] = {}
    node_to_token_positions: Dict[str, set] = {}
    node_to_trunc: Dict[str, int] = {}
    for nid, rec in records.items():
        # Each node represents exactly one value
        if rec.value_index >= 0:
            node_to_value_indices[nid] = [rec.value_index]
            token_range = value_token_ranges_by_index.get(rec.value_index)
            if token_range is not None:
                start, end = token_range
                if end > start:
                    node_to_token_positions[nid] = set(range(start, end))
                else:
                    node_to_token_positions[nid] = set()
            else:
                node_to_token_positions[nid] = set()
        else:
            # Tail node
            node_to_value_indices[nid] = []
            node_to_token_positions[nid] = set()
        
        node_to_trunc[nid] = rec.truncation_token_index

    # Include values that have aligned token spans but no patching record (e.g.,
    # prompt values). These can still contribute as parent-only nodes.
    for vidx, token_range in value_token_ranges_by_index.items():
        if vidx in excluded_value_indices:
            continue
        nid = f"v{vidx}"
        if nid in node_to_value_indices:
            continue
        start, end = token_range
        if end <= start:
            continue
        node_to_value_indices[nid] = [vidx]
        node_to_token_positions[nid] = set(range(start, end))
        node_to_trunc[nid] = int(start)

    if "v0" not in records:
        # If v0 does not exist, use the node with largest truncation index as root.
        root = max(records.values(), key=lambda r: r.truncation_token_index).node_id
    else:
        root = "v0"

    visited = set()
    frontier: List[Tuple[str, int]] = [(root, 0)]

    edges: Dict[Tuple[str, str], float] = {}
    node_stats: Dict[str, Dict] = {}

    while frontier:
        child, depth = frontier.pop(0)
        if child in visited:
            continue
        visited.add(child)
        if depth > cfg.max_depth:
            continue

        rec = records.get(child)
        if rec is None:
            continue

        scores = aggregate_token_scores(rec.diff_delta_matrix, cfg.layer_agg)
        token_idx_local = select_token_positions(
            scores=scores,
            method=cfg.selection_method,
            top_k=cfg.top_k,
            quantile=cfg.quantile,
            fdr_q=cfg.fdr_q,
            min_tokens=cfg.min_tokens,
        )

        child_trunc = rec.truncation_token_index

        # Exclude the token immediately before truncation since it directly predicts
        # the truncated token and can dominate with a mechanism unrelated to parent value reuse.
        selected_pairs: List[Tuple[int, float]] = []
        for i in token_idx_local:
            if not (0 <= i < len(rec.position_indices)):
                continue
            pos = rec.position_indices[i]
            if pos == child_trunc - 1:
                continue
            selected_pairs.append((pos, float(abs(scores[i]))))

        selected_token_positions = [p for p, _ in selected_pairs]
        selected_token_scores = [w for _, w in selected_pairs]

        # Parent candidates must occur earlier in sequence than child.
        parent_nodes = [nid for nid, t in node_to_trunc.items() if t < child_trunc and nid != child]

        # Aggregate token evidence into parent edge weights using only tokens that
        # lie within each parent node's own value token span(s).
        parent_bucket: Dict[str, List[float]] = {}
        for pnode in parent_nodes:
            parent_positions = node_to_token_positions.get(pnode, set())
            if not parent_positions:
                continue
            weights = [w for pos, w in selected_pairs if pos in parent_positions]
            if weights:
                parent_bucket[pnode] = weights

        if parent_bucket:
            parent_strength = {p: float(np.max(ws)) for p, ws in parent_bucket.items()}
            max_w = max(parent_strength.values())
            keep_thresh = cfg.relative_edge_threshold * max_w
            parent_strength = {p: w for p, w in parent_strength.items() if w >= keep_thresh}
        else:
            parent_strength = {}

        for pnode, weight in parent_strength.items():
            edges[(pnode, child)] = weight
            if pnode not in visited:
                frontier.append((pnode, depth + 1))

        node_stats[child] = {
            "truncation_token_index": rec.truncation_token_index,
            "n_token_positions": len(rec.position_indices),
            "n_selected_tokens": len(selected_token_positions),
            "selected_token_positions": selected_token_positions,
            "selected_token_scores_abs": selected_token_scores,
            "mapped_parents": sorted(list(parent_strength.keys())),
        }

    # Add stats for nodes never expanded but present.
    for nid, rec in records.items():
        if nid not in node_stats:
            node_stats[nid] = {
                "truncation_token_index": rec.truncation_token_index,
                "n_token_positions": len(rec.position_indices),
                "n_selected_tokens": 0,
                "selected_token_positions": [],
                "selected_token_scores_abs": [],
                "mapped_parents": [],
            }

    # Add placeholder stats for parent-only synthetic nodes.
    for nid, trunc in node_to_trunc.items():
        if nid in node_stats:
            continue
        node_stats[nid] = {
            "truncation_token_index": int(trunc),
            "n_token_positions": 0,
            "n_selected_tokens": 0,
            "selected_token_positions": [],
            "selected_token_scores_abs": [],
            "mapped_parents": [],
        }

    graph_nodes = []
    for nid, trunc in sorted(node_to_trunc.items(), key=lambda kv: kv[1]):
        rec = records.get(nid)
        if rec is not None:
            value_index = rec.value_index
        else:
            vidxs = node_to_value_indices.get(nid, [])
            value_index = vidxs[0] if vidxs else -1

        vidxs = node_to_value_indices.get(nid, [])
        value_texts = [value_label_by_index.get(idx, "") for idx in vidxs if idx in value_label_by_index]

        graph_nodes.append(
            {
                "id": nid,
                "label": ("v" + str(value_index) if value_index >= 0 else nid),
                "truncation_token_index": int(trunc),
                "value_index": value_index,
                "value_indices": vidxs,
                "value_texts": value_texts,
            }
        )

    graph_edges = [
        {
            "source": src,
            "target": dst,
            "weight": float(w),
        }
        for (src, dst), w in sorted(edges.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    ]

    return {
        "root": root,
        "nodes": graph_nodes,
        "edges": graph_edges,
        "node_stats": node_stats,
        "excluded_shared_values": excluded_shared_value_metadata,
    }


def _dot_escape(text: str) -> str:
    # Preserve DOT newline escapes ("\\n") while escaping quotes and raw newlines.
    return text.replace('"', '\\"').replace("\n", "\\n")


def to_dot(pair_name: str, graph: Dict, trace_text: Optional[str] = None, truncation_labels_text: Optional[str] = None) -> str:
    lines = [f"digraph {pair_name} {{", "  rankdir=RL;"]
    if trace_text or truncation_labels_text:
        label_parts: List[str] = []
        if trace_text:
            compact_trace = " ".join(str(trace_text).split())
            clipped = compact_trace[:600] + ("..." if len(compact_trace) > 600 else "")
            label_parts.append(f"Base Trace: {clipped}")
        if truncation_labels_text:
            label_parts.append("")
            label_parts.append(truncation_labels_text)
        dot_label = "\\n".join(label_parts)
        lines.extend(
            [
                '  labelloc="b";',
                '  labeljust="l";',
                f'  label="{_dot_escape(dot_label)}";',
            ]
        )
    for node in graph["nodes"]:
        nid = node["id"]
        trunc = node["truncation_token_index"]
        value_indices = node.get("value_indices", [])
        value_texts = node.get("value_texts", [])
        vlabel = ",".join([f"v{int(i)}" for i in value_indices]) if value_indices else nid
        tlabel = " | ".join([str(v) for v in value_texts[:2]])
        if len(value_texts) > 2:
            tlabel += " | ..."
        if tlabel:
            label = f"{vlabel}\\n{tlabel}\\n(t={trunc})"
        else:
            label = f"{vlabel}\\n(t={trunc})"
        lines.append(f'  "{nid}" [label="{_dot_escape(label)}"];')
    for edge in graph["edges"]:
        src = edge["source"]
        dst = edge["target"]
        w = edge["weight"]
        lines.append(f'  "{src}" -> "{dst}" [label="{w:.3f}"];')
    lines.append("}")
    return "\n".join(lines)


def _render_with_graphviz(dot_file: Path, out_file: Path, fmt: str, layout: str) -> bool:
    if shutil.which(layout) is None:
        return False
    cmd = [layout, f"-T{fmt}", str(dot_file), "-o", str(out_file)]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def _render_with_matplotlib(
    graph: Dict,
    out_file: Path,
    dpi: int,
    trace_text: Optional[str] = None,
    value_alignment_text: Optional[str] = None,
) -> bool:
    try:
        import matplotlib.pyplot as plt
        from textwrap import wrap
    except Exception:
        return False

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not nodes:
        return False

    # Deterministic lightweight layout:
    # x by reverse truncation order (right-to-left causal flow),
    # y by stable node ordering within the same truncation.
    trunc_groups: Dict[int, List[str]] = {}
    for n in nodes:
        trunc_groups.setdefault(int(n.get("truncation_token_index", 0)), []).append(str(n["id"]))

    sorted_trunc = sorted(trunc_groups.keys(), reverse=True)
    pos: Dict[str, Tuple[float, float]] = {}
    for xi, trunc in enumerate(sorted_trunc):
        group = sorted(trunc_groups[trunc])
        m = len(group)
        for yi, nid in enumerate(group):
            y = (m - 1) / 2.0 - yi
            pos[nid] = (float(xi), float(y))

    # Determine figure size; add space for trace if present
    fig_height = max(4, len(nodes) * 0.8)
    if trace_text or value_alignment_text:
        fig_height += 2.2  # Extra space for trace display
    
    fig, ax = plt.subplots(figsize=(10, fig_height))

    weights = [float(e.get("weight", 1.0)) for e in edges]
    max_w = max(weights) if weights else 1.0
    if max_w <= 0:
        max_w = 1.0

    for e in edges:
        src = str(e["source"])
        dst = str(e["target"])
        if src not in pos or dst not in pos:
            continue
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        w = float(e.get("weight", 0.0))
        lw = 1.0 + 4.0 * (w / max_w)
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "lw": lw, "alpha": 0.8},
        )
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        ax.text(mx, my + 0.08, f"{w:.2f}", fontsize=8, ha="center")

    for n in nodes:
        nid = str(n["id"])
        trunc = int(n.get("truncation_token_index", 0))
        value_indices = n.get("value_indices", [])
        value_texts = n.get("value_texts", [])
        vlabel = ",".join([f"v{int(i)}" for i in value_indices]) if value_indices else nid
        value_line = " | ".join([str(v) for v in value_texts[:2]])
        if len(value_texts) > 2:
            value_line += " | ..."
        x, y = pos[nid]
        ax.scatter([x], [y], s=420, zorder=3)
        if value_line:
            node_text = f"{vlabel}\n{value_line}\nt={trunc}"
        else:
            node_text = f"{vlabel}\nt={trunc}"
        ax.text(x, y, node_text, fontsize=8, ha="center", va="center", color="white", zorder=4)

    ax.set_title("Value Causal Graph")
    ax.set_axis_off()
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
    
    # Add base trace text at bottom if available
    if trace_text or value_alignment_text:
        footer_parts: List[str] = []
        if trace_text:
            footer_parts.append(f"Base Trace: {trace_text}")
        if value_alignment_text:
            footer_parts.append("")
            footer_parts.append(value_alignment_text)
        footer_text = "\n".join(footer_parts)
        fig.text(
            0.5, 0.02,
            footer_text,
            ha="center",
            va="bottom",
            fontsize=7,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )
    
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def render_graph(
    pair_name: str,
    graph: Dict,
    dot_file: Path,
    out_dir: Path,
    fmt: str,
    layout: str,
    dpi: int,
    trace_text: Optional[str] = None,
    value_alignment_text: Optional[str] = None,
) -> Optional[Path]:
    if fmt == "none":
        return None

    out_file = out_dir / f"graph.{fmt}"
    if _render_with_graphviz(dot_file=dot_file, out_file=out_file, fmt=fmt, layout=layout):
        return out_file

    # Fallback only supports png.
    if fmt == "png":
        ok = _render_with_matplotlib(
            graph=graph,
            out_file=out_file,
            dpi=dpi,
            trace_text=trace_text,
            value_alignment_text=value_alignment_text,
        )
        if ok:
            return out_file

    return None


def normalize_pair_dir_name(raw: str) -> str:
    if raw.startswith("pair"):
        return raw
    return f"pair{raw}"


def main() -> None:
    args = parse_args()

    patch_runs_dir: Path = args.patch_runs_dir
    output_dir: Path = args.output_dir or (patch_runs_dir / "graphs")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = GraphBuildConfig(
        layer_agg=args.layer_agg,
        selection_method=args.selection_method,
        top_k=args.top_k,
        quantile=args.quantile,
        fdr_q=args.fdr_q,
        min_tokens=max(1, args.min_tokens),
        relative_edge_threshold=max(0.0, args.relative_edge_threshold),
        max_depth=max(1, args.max_depth),
    )

    if not patch_runs_dir.exists():
        raise FileNotFoundError(f"patch-runs directory not found: {patch_runs_dir}")

    # Load aligned pairs from parent directory (if available)
    aligned_pairs_path = patch_runs_dir.parent / "aligned_pairs.json"
    aligned_pairs_dict = load_aligned_pairs(aligned_pairs_path)
    if aligned_pairs_dict:
        print(f"Loaded {len(aligned_pairs_dict)} pair traces from {aligned_pairs_path}")

    pair_dirs = [p for p in sorted(patch_runs_dir.glob("pair*")) if p.is_dir()]
    if args.pair_id is not None:
        wanted = normalize_pair_dir_name(args.pair_id)
        pair_dirs = [p for p in pair_dirs if p.name == wanted]

    if not pair_dirs:
        print("No pair directories found. Nothing to do.")
        return

    summary = {
        "schema_version": "v1",
        "config": {
            "patch_runs_dir": str(patch_runs_dir),
            "layer_agg": cfg.layer_agg,
            "selection_method": cfg.selection_method,
            "top_k": cfg.top_k,
            "quantile": cfg.quantile,
            "fdr_q": cfg.fdr_q,
            "min_tokens": cfg.min_tokens,
            "relative_edge_threshold": cfg.relative_edge_threshold,
            "max_depth": cfg.max_depth,
        },
        "pairs": [],
    }

    for pair_dir in pair_dirs:
        pair_name = pair_dir.name
        records = read_pair_matrices(pair_dir)
        if not records:
            print(f"[{pair_name}] No matrix records found, skipping.")
            continue

        pair_id = pair_name.replace("pair", "")
        pair_record = get_pair_record(pair_id, aligned_pairs_dict)
        value_label_by_index = build_value_label_by_index(pair_record)
        value_token_ranges_by_index = build_value_token_ranges_by_index(pair_record)
        excluded_value_indices = build_excluded_shared_value_indices(pair_record)
        excluded_shared_value_metadata = build_excluded_shared_value_metadata(pair_record)

        graph = build_graph_for_pair(
            records=records,
            cfg=cfg,
            value_label_by_index=value_label_by_index,
            value_token_ranges_by_index=value_token_ranges_by_index,
            excluded_value_indices=excluded_value_indices,
            excluded_shared_value_metadata=excluded_shared_value_metadata,
        )

        pair_out_dir = output_dir / pair_name
        pair_out_dir.mkdir(parents=True, exist_ok=True)

        graph_json = pair_out_dir / "graph.json"
        dot_file = pair_out_dir / "graph.dot"

        trace_text = get_pair_trace(pair_id, aligned_pairs_dict)
        value_alignment_text = build_value_alignment_line(pair_record, trace_text or "")

        if trace_text:
            print(f"[{pair_name}] trace text loaded ({len(trace_text)} chars)")
        else:
            print(f"[{pair_name}] trace text missing")

        with open(graph_json, "w") as f:
            json.dump(graph, f, indent=2)
        with open(dot_file, "w") as f:
            f.write(to_dot(pair_name, graph, trace_text=trace_text, truncation_labels_text=value_alignment_text))

        rendered = render_graph(
            pair_name=pair_name,
            graph=graph,
            dot_file=dot_file,
            out_dir=pair_out_dir,
            fmt=args.render,
            layout=args.layout,
            dpi=max(72, args.dpi),
            trace_text=trace_text,
            value_alignment_text=value_alignment_text,
        )

        summary["pairs"].append(
            {
                "pair": pair_name,
                "n_nodes": len(graph["nodes"]),
                "n_edges": len(graph["edges"]),
                "n_excluded_shared_values": len(graph.get("excluded_shared_values", [])),
                "root": graph["root"],
                "graph_json": str(graph_json),
                "graph_dot": str(dot_file),
                "graph_render": str(rendered) if rendered is not None else None,
            }
        )

        print(f"[{pair_name}] nodes={len(graph['nodes'])}, edges={len(graph['edges'])}")
        print(f"  wrote {graph_json}")
        print(f"  wrote {dot_file}")
        if rendered is not None:
            print(f"  wrote {rendered}")
        elif args.render != "none":
            print("  render skipped (Graphviz unavailable and fallback unsupported for chosen format)")

    summary_file = output_dir / "graph_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {summary_file}")


if __name__ == "__main__":
    main()
