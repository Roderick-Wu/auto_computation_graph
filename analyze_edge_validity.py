#!/usr/bin/env python3
"""
Edge validity analysis: compare causal score distributions for recovered vs.
non-recovered edges, tagged against the GT graph.

For each (parent, child) pair that is temporally valid, we compute the maximum
causal score over that parent's token span from the child's patching matrix.
We then label each candidate edge as:
  - TP : in patch graph AND in GT graph
  - FP : in patch graph, NOT in GT graph
  - FN : NOT in patch graph, in GT graph
  - TN : NOT in patch graph, NOT in GT graph

Outputs (under --output-dir/<model-name>/):
  - edge_validity_scores.csv    : one row per candidate edge
  - edge_validity_summary.json  : mean/median by status + Mann-Whitney U test
  - edge_validity_plot.png      : violin plot of score by TP/FP/FN/TN

Usage
-----
python analyze_edge_validity.py --model-name Qwen2.5-32B
python analyze_edge_validity.py --model-name Qwen2.5-32B --experiments velocity_from_ke

No GPU required.  Runs in ~15-30 min total across all models.
Requires: numpy, scipy, matplotlib (all available via scipy-stack module on CC clusters).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workspace_paths import resolve_auto_traces_root
from construct_graph import (
    aggregate_token_scores,
    build_excluded_shared_value_indices,
    get_pair_record,
    load_aligned_pairs,
    read_pair_matrices,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_graph_edges(graph_json: Path) -> Set[Tuple[str, str]]:
    if not graph_json.exists():
        return set()
    with open(graph_json) as f:
        obj = json.load(f)
    return {(e["source"], e["target"]) for e in obj.get("edges", [])}


def load_graph_nodes(graph_json: Path) -> List[Dict]:
    if not graph_json.exists():
        return []
    with open(graph_json) as f:
        obj = json.load(f)
    return obj.get("nodes", [])


def nodes_in_token_order(nodes: List[Dict]) -> List[str]:
    return [
        n["id"]
        for n in sorted(nodes, key=lambda n: int(n.get("truncation_token_index", 0)))
    ]


def compute_parent_max_scores(
    pair_matrices_dir: Path,
    nodes: List[Dict],
    excluded_indices: Set[int],
    layer_agg: str,
) -> Dict[Tuple[str, str], float]:
    """
    For every temporally valid (parent, child) pair, return the maximum
    causal score over parent-span tokens in the child's patching matrix.

    Parent span is approximated as [trunc(parent), trunc(next_node)) — this is
    an approximation; the exact span is in aligned_pairs.json but this avoids
    an extra load per pair and is sufficient for distribution comparison.

    Returns {(parent_id, child_id): max_score}.  Returns 0.0 when the parent
    span contains no patched positions (parent token positions not covered).
    """
    records = read_pair_matrices(pair_matrices_dir)
    if not records:
        return {}

    node_to_trunc: Dict[str, int] = {
        n["id"]: int(n.get("truncation_token_index", 0)) for n in nodes
    }
    ordered_ids = nodes_in_token_order(nodes)

    out: Dict[Tuple[str, str], float] = {}

    for child_id, rec in records.items():
        if child_id not in node_to_trunc:
            continue
        if rec.value_index in excluded_indices:
            continue

        scores = aggregate_token_scores(rec.diff_delta_matrix, layer_agg)
        child_trunc = node_to_trunc[child_id]

        parent_candidates = [
            nid for nid in ordered_ids
            if node_to_trunc.get(nid, 0) < child_trunc and nid != child_id
        ]

        for pidx, parent_id in enumerate(ordered_ids):
            if parent_id not in parent_candidates:
                continue

            # Approximate span: [trunc(parent), trunc(next_node_in_order))
            pos_in_ordered = ordered_ids.index(parent_id)
            span_start = node_to_trunc[parent_id]
            if pos_in_ordered + 1 < len(ordered_ids):
                span_end = node_to_trunc[ordered_ids[pos_in_ordered + 1]]
            else:
                span_end = child_trunc

            parent_scores = [
                float(abs(scores[local_i]))
                for local_i, pos in enumerate(rec.position_indices)
                if span_start <= pos < span_end and local_i < len(scores)
            ]

            out[(parent_id, child_id)] = float(np.max(parent_scores)) if parent_scores else 0.0

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", required=True)
    p.add_argument("--experiments", default="",
                   help="Comma-separated; default = all experiments for this model.")
    p.add_argument("--graphs-subdir", default="graphs")
    p.add_argument("--gt-subdir", default="graphs_ground_truth_api")
    p.add_argument("--output-dir", type=Path, default=Path("smoke_logs/edge_validity"))
    p.add_argument("--layer-agg", default="mean_abs",
                   choices=["mean_abs", "max_abs", "mean_signed"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = args.output_dir / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    placeholder_root = resolve_auto_traces_root(args.model_name, "__placeholder__")
    model_root = placeholder_root.parent

    requested = [x.strip() for x in args.experiments.split(",") if x.strip()] or None
    experiments = (
        requested
        if requested
        else sorted(p.name for p in model_root.iterdir() if p.is_dir())
    )

    all_rows: List[Dict] = []

    for exp in experiments:
        exp_root = resolve_auto_traces_root(args.model_name, exp)
        patch_root = exp_root / args.graphs_subdir
        gt_root = exp_root / args.gt_subdir
        patch_runs_root = exp_root / "patch_runs"
        aligned_pairs_path = exp_root / "aligned_pairs.json"

        if not patch_root.exists() or not patch_runs_root.exists():
            print(f"[skip] {exp}: missing graphs or patch_runs dir")
            continue

        aligned_pairs = load_aligned_pairs(aligned_pairs_path)

        pair_dirs = sorted(d for d in patch_root.glob("pair*") if d.is_dir())
        for pair_dir in pair_dirs:
            pair_name = pair_dir.name
            pair_id_str = pair_name.replace("pair", "")

            patch_graph_json = pair_dir / "graph.json"
            gt_graph_json = gt_root / pair_name / "graph.json"
            pair_matrices_dir = patch_runs_root / pair_name

            if not patch_graph_json.exists() or not gt_graph_json.exists():
                continue
            if not pair_matrices_dir.exists():
                continue

            nodes = load_graph_nodes(patch_graph_json)
            if not nodes:
                continue

            patch_edges = load_graph_edges(patch_graph_json)
            gt_edges = load_graph_edges(gt_graph_json)

            pair_record = get_pair_record(pair_id_str, aligned_pairs)
            excluded = build_excluded_shared_value_indices(pair_record) if pair_record else set()

            parent_scores = compute_parent_max_scores(
                pair_matrices_dir, nodes, excluded, args.layer_agg
            )

            for (parent_id, child_id), score in parent_scores.items():
                in_patch = (parent_id, child_id) in patch_edges
                in_gt = (parent_id, child_id) in gt_edges
                if in_patch and in_gt:
                    status = "TP"
                elif in_patch and not in_gt:
                    status = "FP"
                elif not in_patch and in_gt:
                    status = "FN"
                else:
                    status = "TN"

                all_rows.append({
                    "model": args.model_name,
                    "experiment": exp,
                    "pair": pair_name,
                    "parent": parent_id,
                    "child": child_id,
                    "status": status,
                    "max_score": score,
                })

        print(f"[done] {exp}  ({len(pair_dirs)} pairs, {len(all_rows)} candidate edges so far)")

    if not all_rows:
        print("No rows collected — check paths.")
        return

    # Write CSV
    csv_path = out_dir / "edge_validity_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows → {csv_path}")

    # Summary statistics + Mann-Whitney U
    by_status: Dict[str, List[float]] = {}
    for row in all_rows:
        by_status.setdefault(row["status"], []).append(row["max_score"])

    summary: Dict = {}
    for status, scores_list in sorted(by_status.items()):
        arr = np.array(scores_list)
        summary[status] = {
            "n": int(len(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
        }

    # Key test: recovered edges (TP + FP) vs non-recovered (TN + FN)
    edge_scores = by_status.get("TP", []) + by_status.get("FP", [])
    non_edge_scores = by_status.get("TN", []) + by_status.get("FN", [])
    if edge_scores and non_edge_scores:
        try:
            from scipy import stats as scipy_stats
            u_stat, p_val = scipy_stats.mannwhitneyu(
                edge_scores, non_edge_scores, alternative="greater"
            )
            summary["mann_whitney_edges_vs_nonedges"] = {
                "n_edges": len(edge_scores),
                "n_nonedges": len(non_edge_scores),
                "U": float(u_stat),
                "p_value": float(p_val),
                "significant_at_0.05": bool(p_val < 0.05),
                "interpretation": (
                    "Recovered edges have significantly higher causal scores than non-edges (p < 0.05)"
                    if p_val < 0.05
                    else "Difference not significant at p=0.05"
                ),
            }
        except ImportError:
            summary["mann_whitney_edges_vs_nonedges"] = {"error": "scipy not available"}

    json_path = out_dir / "edge_validity_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {json_path}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Violin plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = ["TP", "FP", "FN", "TN"]
        labels = [s for s in order if s in by_status]
        data = [by_status[s] for s in labels]

        fig, ax = plt.subplots(figsize=(7, 4))
        parts = ax.violinplot(data, showmedians=True, showquartiles=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_xlabel("Edge status vs. GT graph", fontsize=12)
        ax.set_ylabel("Max causal score (|Δ_diff|)", fontsize=12)
        ax.set_title(f"Causal score distributions — {args.model_name}", fontsize=13)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        plot_path = out_dir / "edge_validity_plot.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Plot → {plot_path}")
    except ImportError:
        print("matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()
