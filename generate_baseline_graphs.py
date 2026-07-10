#!/usr/bin/env python3
"""
Generate structural baseline graphs for all pairs in a model/experiment.

Baselines (each writes to its own subdir under the experiment root):
  - random_dag     : random temporal DAG (edges sampled uniformly, respecting
                     token order); averaged over n_seeds for reporting
  - nearest_parent : each node's sole parent is the immediately preceding node
  - two_parents    : each node's parents are the up-to-two preceding nodes
  - prompt_only    : every CoT node's parents are all prompt-region nodes;
                     no CoT-to-CoT edges

Usage
-----
python generate_baseline_graphs.py \\
    --model-name Qwen2.5-32B \\
    --experiments velocity_from_ke,current_from_power \\
    --baselines random_dag,nearest_parent,two_parents,prompt_only \\
    --random-edge-prob 0.3 \\
    --n-random-seeds 5

Output directories (under each experiment root):
    graphs_baseline_nearest_parent/pair<N>/graph.json
    graphs_baseline_two_parents/pair<N>/graph.json
    graphs_baseline_prompt_only/pair<N>/graph.json
    graphs_baseline_random_dag_seed<K>/pair<N>/graph.json   (one dir per seed)

Each graph.json matches the schema used by evaluate_graphs_against_ground_truth.py:
    {
        "root": "v0",
        "nodes": [{"id": "vN", "truncation_token_index": T}, ...],
        "edges": [{"source": "vA", "target": "vB", "weight": 1.0}, ...]
    }

No GPU required.  Runs in < 5 min on all models/experiments.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workspace_paths import resolve_auto_traces_root


# ---------------------------------------------------------------------------
# Graph I/O helpers
# ---------------------------------------------------------------------------

def load_patch_graph(graph_json: Path) -> Tuple[List[Dict], List[Dict], Optional[int]]:
    """Return (nodes, edges, prompt_token_cutoff) from a patching graph.json."""
    if not graph_json.exists():
        return [], [], None
    with open(graph_json) as f:
        obj = json.load(f)
    return (
        obj.get("nodes", []),
        obj.get("edges", []),
        obj.get("prompt_token_cutoff"),
    )


def write_graph(path: Path, nodes: List[Dict], edges: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = nodes[0]["id"] if nodes else "v0"
    with open(path, "w") as f:
        json.dump({"root": root, "nodes": nodes, "edges": edges}, f, indent=2)


def nodes_in_token_order(nodes: List[Dict]) -> List[str]:
    """Return node IDs sorted ascending by truncation_token_index."""
    return [
        n["id"]
        for n in sorted(nodes, key=lambda n: int(n.get("truncation_token_index", 0)))
    ]


def prompt_node_ids(nodes: List[Dict], prompt_token_cutoff: Optional[int]) -> Set[str]:
    """Return IDs of nodes whose truncation_token_index < prompt_token_cutoff."""
    if prompt_token_cutoff is None:
        return set()
    return {
        n["id"]
        for n in nodes
        if int(n.get("truncation_token_index", 0)) < prompt_token_cutoff
    }


# ---------------------------------------------------------------------------
# Baseline constructors
# ---------------------------------------------------------------------------

def make_random_dag(ordered_ids: List[str], edge_prob: float, seed: int) -> List[Dict]:
    """Randomly sample edges from temporally valid pairs (earlier → later).

    Each non-root node is guaranteed at least one parent to avoid isolated nodes.
    """
    rng = random.Random(seed)
    edges: List[Dict] = []
    for i, child in enumerate(ordered_ids):
        has_parent = False
        for parent in ordered_ids[:i]:
            if rng.random() < edge_prob:
                edges.append({"source": parent, "target": child, "weight": 1.0})
                has_parent = True
        # Guarantee connectivity: if no parent was sampled, attach the predecessor.
        if i > 0 and not has_parent:
            edges.append({"source": ordered_ids[i - 1], "target": child, "weight": 1.0})
    return edges


def make_nearest_parent(ordered_ids: List[str]) -> List[Dict]:
    """Chain baseline: each node's sole parent is the immediately preceding node."""
    return [
        {"source": ordered_ids[i - 1], "target": ordered_ids[i], "weight": 1.0}
        for i in range(1, len(ordered_ids))
    ]


def make_two_parents(ordered_ids: List[str]) -> List[Dict]:
    """Two-hop chain baseline: each node's parents are the up-to-two preceding nodes."""
    edges: List[Dict] = []
    for i in range(1, len(ordered_ids)):
        for j in range(max(0, i - 2), i):
            edges.append({"source": ordered_ids[j], "target": ordered_ids[i], "weight": 1.0})
    return edges


def make_prompt_only(ordered_ids: List[str], prompt_ids: Set[str]) -> List[Dict]:
    """Every CoT node's parents are all prompt-region nodes; no CoT-to-CoT edges.

    Falls back to the first node as sole parent if no prompt nodes are detected.
    """
    if not prompt_ids:
        # No prompt cutoff info available — use the very first node as fallback root.
        fallback = ordered_ids[0]
        return [
            {"source": fallback, "target": child, "weight": 1.0}
            for child in ordered_ids[1:]
        ]
    edges: List[Dict] = []
    for nid in ordered_ids:
        if nid in prompt_ids:
            continue  # prompt nodes are roots — no incoming edges
        for pid in ordered_ids:
            if pid in prompt_ids:
                edges.append({"source": pid, "target": nid, "weight": 1.0})
    return edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BASELINE_NAMES = ["random_dag", "nearest_parent", "two_parents", "prompt_only"]


def process_pair(
    pair_dir: Path,
    gt_pair_dir: Path,
    exp_root: Path,
    baselines: List[str],
    edge_prob: float,
    n_seeds: int,
) -> None:
    patch_graph_json = pair_dir / "graph.json"
    gt_graph_json = gt_pair_dir / "graph.json"

    if not patch_graph_json.exists() or not gt_graph_json.exists():
        return

    nodes, _, prompt_cutoff = load_patch_graph(patch_graph_json)
    if not nodes:
        return

    ordered_ids = nodes_in_token_order(nodes)
    if len(ordered_ids) < 2:
        return

    p_ids = prompt_node_ids(nodes, prompt_cutoff)
    pair_name = pair_dir.name  # e.g. "pair42"

    for baseline in baselines:
        if baseline == "random_dag":
            for seed in range(n_seeds):
                edges = make_random_dag(ordered_ids, edge_prob, seed=seed)
                out_path = exp_root / f"graphs_baseline_random_dag_seed{seed}" / pair_name / "graph.json"
                write_graph(out_path, nodes, edges)
        elif baseline == "nearest_parent":
            edges = make_nearest_parent(ordered_ids)
            write_graph(exp_root / "graphs_baseline_nearest_parent" / pair_name / "graph.json", nodes, edges)
        elif baseline == "two_parents":
            edges = make_two_parents(ordered_ids)
            write_graph(exp_root / "graphs_baseline_two_parents" / pair_name / "graph.json", nodes, edges)
        elif baseline == "prompt_only":
            edges = make_prompt_only(ordered_ids, p_ids)
            write_graph(exp_root / "graphs_baseline_prompt_only" / pair_name / "graph.json", nodes, edges)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", required=True)
    p.add_argument("--experiments", default="",
                   help="Comma-separated list; default = all experiments under traces/<model>.")
    p.add_argument("--baselines", default=",".join(BASELINE_NAMES),
                   help=f"Comma-separated subset of: {', '.join(BASELINE_NAMES)}")
    p.add_argument("--random-edge-prob", type=float, default=0.3,
                   help="Per-pair edge probability for random_dag baseline (default: 0.3).")
    p.add_argument("--n-random-seeds", type=int, default=5,
                   help="Number of random seeds for random_dag baseline (default: 5).")
    p.add_argument("--graphs-subdir", default="graphs",
                   help="Subdirectory containing the patching graphs (default: graphs).")
    p.add_argument("--gt-subdir", default="graphs_ground_truth_api",
                   help="Subdirectory containing GT graphs (default: graphs_ground_truth_api).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    baselines = [x.strip() for x in args.baselines.split(",") if x.strip()]
    unknown = [b for b in baselines if b not in BASELINE_NAMES]
    if unknown:
        print(f"WARNING: unknown baselines ignored: {unknown}")
        baselines = [b for b in baselines if b in BASELINE_NAMES]

    # Resolve experiment list
    placeholder_root = resolve_auto_traces_root(args.model_name, "__placeholder__")
    model_root = placeholder_root.parent

    requested = [x.strip() for x in args.experiments.split(",") if x.strip()] or None
    experiments = (
        requested
        if requested
        else sorted(p.name for p in model_root.iterdir() if p.is_dir())
    )

    total_pairs = 0
    for exp in experiments:
        exp_root = resolve_auto_traces_root(args.model_name, exp)
        patch_root = exp_root / args.graphs_subdir
        gt_root = exp_root / args.gt_subdir

        if not patch_root.exists():
            print(f"[skip] no graphs dir: {patch_root}")
            continue

        pair_dirs = sorted(d for d in patch_root.glob("pair*") if d.is_dir())
        n = 0
        for pair_dir in pair_dirs:
            gt_pair_dir = gt_root / pair_dir.name
            process_pair(
                pair_dir=pair_dir,
                gt_pair_dir=gt_pair_dir,
                exp_root=exp_root,
                baselines=baselines,
                edge_prob=args.random_edge_prob,
                n_seeds=args.n_random_seeds,
            )
            n += 1
        total_pairs += n
        print(f"[done] {args.model_name}/{exp}  ({n} pairs)")

    print(f"\nTotal pairs processed: {total_pairs}")
    print("Baselines written:", baselines)
    if "random_dag" in baselines:
        print(f"  random_dag: {args.n_random_seeds} seeds × edge_prob={args.random_edge_prob}")


if __name__ == "__main__":
    main()
