#!/usr/bin/env python3
"""
Evaluate candidate causal graphs against API-generated ground-truth graphs.

Default comparison:
  candidate: /scratch/<user>/traces/<model>/<experiment>/graphs
  ground-truth: /scratch/<user>/traces/<model>/<experiment>/graphs_ground_truth_api

Outputs:
  - pair_metrics.tsv
  - experiment_metrics.tsv
  - overall_metrics.tsv
  - summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workspace_paths import resolve_auto_traces_root, resolve_scratch_root


Edge = Tuple[str, str]


@dataclass
class PRFCounts:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> Optional[float]:
        d = self.tp + self.fp
        return None if d == 0 else self.tp / d

    @property
    def recall(self) -> Optional[float]:
        d = self.tp + self.fn
        return None if d == 0 else self.tp / d

    @property
    def f1(self) -> Optional[float]:
        p = self.precision
        r = self.recall
        if p is None or r is None:
            return None
        d = p + r
        return None if d == 0 else (2.0 * p * r) / d

    @property
    def jaccard(self) -> Optional[float]:
        d = self.tp + self.fp + self.fn
        return None if d == 0 else self.tp / d


def safe_mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def fmt(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{x:.6f}"


def load_graph(graph_json: Path) -> Tuple[Set[str], Set[Edge]]:
    with open(graph_json, "r") as f:
        obj = json.load(f)
    nodes: Set[str] = set()
    for n in obj.get("nodes", []):
        nid = n.get("id")
        if isinstance(nid, str):
            nodes.add(nid)
    edges: Set[Edge] = set()
    for e in obj.get("edges", []):
        s = e.get("source")
        t = e.get("target")
        if isinstance(s, str) and isinstance(t, str):
            edges.add((s, t))
    return nodes, edges


def closure(nodes: Set[str], edges: Set[Edge]) -> Set[Edge]:
    outgoing: Dict[str, List[str]] = defaultdict(list)
    for s, t in edges:
        if s in nodes and t in nodes and s != t:
            outgoing[s].append(t)

    out: Set[Edge] = set()
    for src in nodes:
        stack = list(outgoing.get(src, []))
        seen: Set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur != src:
                out.add((src, cur))
            for nxt in outgoing.get(cur, []):
                if nxt not in seen:
                    stack.append(nxt)
    return out


def prf_counts(pred: Set[Edge], gold: Set[Edge]) -> PRFCounts:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return PRFCounts(tp=tp, fp=fp, fn=fn)


def choose_experiments(model_name: str, requested: Optional[List[str]]) -> List[str]:
    if requested:
        return requested
    root = resolve_scratch_root() / "traces" / model_name
    exps = sorted([p.name for p in root.iterdir() if p.is_dir()])
    return exps


def pair_dirs(path: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not path.exists():
        return out
    for d in sorted(path.glob("pair*")):
        if d.is_dir():
            out[d.name] = d
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare candidate graphs against ground-truth graphs.")
    p.add_argument("--model-name", type=str, default="Qwen2.5-32B")
    p.add_argument(
        "--experiments",
        type=str,
        default="",
        help="Comma-separated experiments; default is all under traces/<model>.",
    )
    p.add_argument("--candidate-subdir", type=str, default="graphs")
    p.add_argument("--ground-truth-subdir", type=str, default="graphs_ground_truth_api")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "smoke_logs" / "graph_eval_qwen_vs_gt",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    requested = [x.strip() for x in args.experiments.split(",") if x.strip()] or None
    experiments = choose_experiments(args.model_name, requested)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_rows: List[Dict[str, object]] = []
    exp_rows: List[Dict[str, object]] = []

    overall_edge = PRFCounts(tp=0, fp=0, fn=0)
    overall_closure = PRFCounts(tp=0, fp=0, fn=0)
    overall_pairs = 0
    overall_skipped = 0

    for exp in experiments:
        exp_root = resolve_auto_traces_root(args.model_name, exp)
        cand_root = exp_root / args.candidate_subdir
        gt_root = exp_root / args.ground_truth_subdir

        cand_pairs = pair_dirs(cand_root)
        gt_pairs = pair_dirs(gt_root)
        shared = sorted(set(cand_pairs) & set(gt_pairs))

        if not shared:
            exp_rows.append(
                {
                    "experiment": exp,
                    "pairs_compared": 0,
                    "pairs_skipped": 0,
                    "edge_precision_micro": None,
                    "edge_recall_micro": None,
                    "edge_f1_micro": None,
                    "edge_jaccard_micro": None,
                    "closure_precision_micro": None,
                    "closure_recall_micro": None,
                    "closure_f1_micro": None,
                    "closure_jaccard_micro": None,
                    "edge_f1_macro": None,
                    "closure_f1_macro": None,
                }
            )
            continue

        exp_edge = PRFCounts(tp=0, fp=0, fn=0)
        exp_closure = PRFCounts(tp=0, fp=0, fn=0)
        exp_edge_f1s: List[Optional[float]] = []
        exp_closure_f1s: List[Optional[float]] = []
        exp_skipped = 0

        for pair_name in shared:
            cand_graph = cand_pairs[pair_name] / "graph.json"
            gt_graph = gt_pairs[pair_name] / "graph.json"
            if not cand_graph.exists() or not gt_graph.exists():
                exp_skipped += 1
                continue

            cand_nodes, cand_edges_raw = load_graph(cand_graph)
            gt_nodes, gt_edges_raw = load_graph(gt_graph)
            common_nodes = cand_nodes & gt_nodes

            cand_edges = {(s, t) for (s, t) in cand_edges_raw if s in common_nodes and t in common_nodes}
            gt_edges = {(s, t) for (s, t) in gt_edges_raw if s in common_nodes and t in common_nodes}

            edge_counts = prf_counts(cand_edges, gt_edges)
            cand_closure = closure(common_nodes, cand_edges)
            gt_closure = closure(common_nodes, gt_edges)
            closure_counts = prf_counts(cand_closure, gt_closure)

            exp_edge.tp += edge_counts.tp
            exp_edge.fp += edge_counts.fp
            exp_edge.fn += edge_counts.fn
            exp_closure.tp += closure_counts.tp
            exp_closure.fp += closure_counts.fp
            exp_closure.fn += closure_counts.fn
            exp_edge_f1s.append(edge_counts.f1)
            exp_closure_f1s.append(closure_counts.f1)

            node_precision = (len(common_nodes) / len(cand_nodes)) if cand_nodes else None
            node_recall = (len(common_nodes) / len(gt_nodes)) if gt_nodes else None
            node_f1 = None
            if node_precision is not None and node_recall is not None and (node_precision + node_recall) > 0:
                node_f1 = 2 * node_precision * node_recall / (node_precision + node_recall)

            pair_rows.append(
                {
                    "experiment": exp,
                    "pair": pair_name,
                    "n_nodes_candidate": len(cand_nodes),
                    "n_nodes_ground_truth": len(gt_nodes),
                    "n_nodes_common": len(common_nodes),
                    "n_edges_candidate_common": len(cand_edges),
                    "n_edges_ground_truth_common": len(gt_edges),
                    "edge_tp": edge_counts.tp,
                    "edge_fp": edge_counts.fp,
                    "edge_fn": edge_counts.fn,
                    "edge_precision": edge_counts.precision,
                    "edge_recall": edge_counts.recall,
                    "edge_f1": edge_counts.f1,
                    "edge_jaccard": edge_counts.jaccard,
                    "closure_tp": closure_counts.tp,
                    "closure_fp": closure_counts.fp,
                    "closure_fn": closure_counts.fn,
                    "closure_precision": closure_counts.precision,
                    "closure_recall": closure_counts.recall,
                    "closure_f1": closure_counts.f1,
                    "closure_jaccard": closure_counts.jaccard,
                    "node_precision": node_precision,
                    "node_recall": node_recall,
                    "node_f1": node_f1,
                }
            )

        overall_edge.tp += exp_edge.tp
        overall_edge.fp += exp_edge.fp
        overall_edge.fn += exp_edge.fn
        overall_closure.tp += exp_closure.tp
        overall_closure.fp += exp_closure.fp
        overall_closure.fn += exp_closure.fn
        overall_pairs += len(shared) - exp_skipped
        overall_skipped += exp_skipped

        exp_rows.append(
            {
                "experiment": exp,
                "pairs_compared": len(shared) - exp_skipped,
                "pairs_skipped": exp_skipped,
                "edge_precision_micro": exp_edge.precision,
                "edge_recall_micro": exp_edge.recall,
                "edge_f1_micro": exp_edge.f1,
                "edge_jaccard_micro": exp_edge.jaccard,
                "closure_precision_micro": exp_closure.precision,
                "closure_recall_micro": exp_closure.recall,
                "closure_f1_micro": exp_closure.f1,
                "closure_jaccard_micro": exp_closure.jaccard,
                "edge_f1_macro": safe_mean(exp_edge_f1s),
                "closure_f1_macro": safe_mean(exp_closure_f1s),
            }
        )

    overall_row = {
        "model_name": args.model_name,
        "candidate_subdir": args.candidate_subdir,
        "ground_truth_subdir": args.ground_truth_subdir,
        "experiments_requested": len(experiments),
        "pairs_compared_total": overall_pairs,
        "pairs_skipped_total": overall_skipped,
        "edge_tp": overall_edge.tp,
        "edge_fp": overall_edge.fp,
        "edge_fn": overall_edge.fn,
        "edge_precision_micro": overall_edge.precision,
        "edge_recall_micro": overall_edge.recall,
        "edge_f1_micro": overall_edge.f1,
        "edge_jaccard_micro": overall_edge.jaccard,
        "closure_tp": overall_closure.tp,
        "closure_fp": overall_closure.fp,
        "closure_fn": overall_closure.fn,
        "closure_precision_micro": overall_closure.precision,
        "closure_recall_micro": overall_closure.recall,
        "closure_f1_micro": overall_closure.f1,
        "closure_jaccard_micro": overall_closure.jaccard,
        "edge_f1_macro_over_experiments": safe_mean([r.get("edge_f1_micro") for r in exp_rows]),
        "closure_f1_macro_over_experiments": safe_mean([r.get("closure_f1_micro") for r in exp_rows]),
    }

    def write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
        if not rows:
            path.write_text("")
            return
        keys = list(rows[0].keys())
        lines = ["\t".join(keys)]
        for r in rows:
            vals: List[str] = []
            for k in keys:
                v = r.get(k)
                if isinstance(v, float):
                    vals.append(fmt(v))
                elif v is None:
                    vals.append("NA")
                else:
                    vals.append(str(v))
            lines.append("\t".join(vals))
        path.write_text("\n".join(lines) + "\n")

    write_tsv(out_dir / "pair_metrics.tsv", pair_rows)
    write_tsv(out_dir / "experiment_metrics.tsv", exp_rows)
    write_tsv(out_dir / "overall_metrics.tsv", [overall_row])

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model_name": args.model_name,
            "experiments": experiments,
            "candidate_subdir": args.candidate_subdir,
            "ground_truth_subdir": args.ground_truth_subdir,
            "output_dir": str(out_dir),
        },
        "overall": overall_row,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {out_dir / 'overall_metrics.tsv'}")
    print(f"Wrote: {out_dir / 'experiment_metrics.tsv'}")
    print(f"Wrote: {out_dir / 'pair_metrics.tsv'}")
    print(f"Wrote: {out_dir / 'summary.json'}")
    print(
        "Overall edge micro: "
        f"P={fmt(overall_edge.precision)} R={fmt(overall_edge.recall)} F1={fmt(overall_edge.f1)} "
        f"J={fmt(overall_edge.jaccard)}"
    )
    print(
        "Overall closure micro: "
        f"P={fmt(overall_closure.precision)} R={fmt(overall_closure.recall)} F1={fmt(overall_closure.f1)} "
        f"J={fmt(overall_closure.jaccard)}"
    )


if __name__ == "__main__":
    main()
