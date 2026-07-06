#!/usr/bin/env python3
"""Run a BH-only FDR-q graph-construction sweep and summarize evaluation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workspace_paths import resolve_auto_traces_root


DEFAULT_MODELS = [
    "gpt-oss-20b",
    "Llama3.1-70B",
    "Mistral-Small-3.1-24B-Base-2503",
]
DEFAULT_Q_VALUES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def q_tag(q: float) -> str:
    return f"{q:g}".replace(".", "p")


def read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def registered_experiments() -> List[str]:
    out = subprocess.check_output(
        [sys.executable, str(REPO_ROOT / "src" / "list_all_experiments.py")],
        cwd=REPO_ROOT,
        text=True,
    )
    return [line.split("\t", 1)[0] for line in out.splitlines() if line.strip()]


def existing_patch_experiments(model: str, experiments: Iterable[str]) -> List[str]:
    out: List[str] = []
    for exp in experiments:
        if (resolve_auto_traces_root(model, exp) / "patch_runs").is_dir():
            out.append(exp)
    return out


def construct_for_experiment(model: str, exp: str, q: float, candidate_subdir: str, log_dir: Path) -> None:
    exp_root = resolve_auto_traces_root(model, exp)
    patch_dir = exp_root / "patch_runs"
    output_dir = exp_root / candidate_subdir
    log_path = log_dir / f"construct_{model}_{exp}_q{q_tag(q)}.log"
    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "src" / "construct_graph.py"),
        "--patch-runs-dir",
        str(patch_dir),
        "--output-dir",
        str(output_dir),
        "--render",
        "none",
        "--layout",
        "dot",
        "--layer-agg",
        "max_abs",
        "--selection-method",
        "fdr",
        "--fdr-q",
        str(q),
        "--min-tokens",
        "0",
        "--relative-edge-threshold",
        "0.0",
        "--parent-causal-rule",
        "bh_only",
        "--edge-build-scope",
        "all_nodes",
        "--strongest-min-weight",
        "0.0",
    ]
    with open(log_path, "w") as log:
        subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def evaluate(model: str, experiments: List[str], candidate_subdir: str, output_dir: Path) -> Dict[str, object]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "evaluate_graphs_against_ground_truth.py"),
        "--model-name",
        model,
        "--experiments",
        ",".join(experiments),
        "--candidate-subdir",
        candidate_subdir,
        "--ground-truth-subdir",
        "graphs_ground_truth_api",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    with open(output_dir / "summary.json") as f:
        return json.load(f)["overall"]


def graph_counts(model: str, experiments: List[str], candidate_subdir: str) -> Dict[str, object]:
    n_graphs = 0
    n_edges = 0
    n_nodes = 0
    for exp in experiments:
        graph_root = resolve_auto_traces_root(model, exp) / candidate_subdir
        for graph_path in sorted(graph_root.glob("pair*/graph.json")):
            with open(graph_path) as f:
                graph = json.load(f)
            n_graphs += 1
            n_edges += len(graph.get("edges", []))
            n_nodes += len(graph.get("nodes", []))
    return {
        "graphs_constructed": n_graphs,
        "avg_edges": (n_edges / n_graphs) if n_graphs else "",
        "avg_nodes": (n_nodes / n_graphs) if n_graphs else "",
    }


def to_float(value: object) -> float:
    if value in ("", None, "NA"):
        return float("nan")
    return float(value)


def make_plots(rows: List[Dict[str, object]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on cluster module set
        print(f"Skipping plots because matplotlib is unavailable: {exc}", file=sys.stderr)
        return

    model_labels = {
        "gpt-oss-20b": "GPT-OSS-20B",
        "Llama3.1-70B": "Llama-3.1-70B",
        "Mistral-Small-3.1-24B-Base-2503": "Mistral-Small-3.1-24B",
    }

    def plot_metric(filename: str, ylabel: str, metric_keys: List[str], labels: List[str]) -> None:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for model in sorted({str(r["model"]) for r in rows}):
            model_rows = sorted([r for r in rows if r["model"] == model], key=lambda r: to_float(r["fdr_q"]))
            xs = [to_float(r["fdr_q"]) for r in model_rows]
            for metric_key, suffix in zip(metric_keys, labels):
                ys = [to_float(r[metric_key]) for r in model_rows]
                label = model_labels.get(model, model)
                if suffix:
                    label = f"{label} {suffix}"
                ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("BH FDR q")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=220)
        plt.close(fig)

    plot_metric("fdr_q_sweep_direct_f1.png", "Direct edge F1", ["edge_f1_micro"], [""])
    plot_metric("fdr_q_sweep_closure_f1.png", "Transitive closure F1", ["closure_f1_micro"], [""])
    plot_metric("fdr_q_sweep_avg_edges.png", "Average candidate edges per graph", ["avg_edges"], [""])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True)
    for ax, prefix, title in [
        (axes[0], "edge", "Direct edges"),
        (axes[1], "closure", "Transitive closure"),
    ]:
        for model in sorted({str(r["model"]) for r in rows}):
            model_rows = sorted([r for r in rows if r["model"] == model], key=lambda r: to_float(r["fdr_q"]))
            xs = [to_float(r["fdr_q"]) for r in model_rows]
            label = model_labels.get(model, model)
            ax.plot(xs, [to_float(r[f"{prefix}_precision_micro"]) for r in model_rows], marker="o", label=f"{label} P")
            ax.plot(xs, [to_float(r[f"{prefix}_recall_micro"]) for r in model_rows], marker="s", linestyle="--", label=f"{label} R")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("BH FDR q")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[1].legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "fdr_q_sweep_precision_recall.png", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    p.add_argument("--fdr-q-values", type=str, default=",".join(str(q) for q in DEFAULT_Q_VALUES))
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "compiled_results" / "fdr_q_sweep")
    p.add_argument("--skip-existing", action="store_true", help="Skip graph construction when an output summary exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    q_values = [float(x.strip()) for x in args.fdr_q_values.split(",") if x.strip()]
    out_dir = args.output_dir.resolve()
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    experiments = registered_experiments()
    overall_rows: List[Dict[str, object]] = []
    exp_rows: List[Dict[str, object]] = []

    for model in models:
        model_exps = existing_patch_experiments(model, experiments)
        print(f"{model}: {len(model_exps)} patched experiments")
        for q in q_values:
            tag = q_tag(q)
            candidate_subdir = f"graphs_bh_only_fdr_q_{tag}"
            eval_dir = REPO_ROOT / "smoke_logs" / f"graph_eval_{model}_bh_fdr_q_{tag}"
            print(f"  q={q:g}: constructing {candidate_subdir}")
            if not (args.skip_existing and (eval_dir / "summary.json").exists()):
                for exp in model_exps:
                    construct_for_experiment(model, exp, q, candidate_subdir, log_dir)
                overall = evaluate(model, model_exps, candidate_subdir, eval_dir)
            else:
                with open(eval_dir / "summary.json") as f:
                    overall = json.load(f)["overall"]

            counts = graph_counts(model, model_exps, candidate_subdir)
            row: Dict[str, object] = {
                "model": model,
                "fdr_q": q,
                "candidate_subdir": candidate_subdir,
                **overall,
                **counts,
            }
            overall_rows.append(row)

            for exp_row in read_tsv(eval_dir / "experiment_metrics.tsv"):
                exp_rows.append({"model": model, "fdr_q": q, "candidate_subdir": candidate_subdir, **exp_row})

    write_csv(out_dir / "fdr_q_sweep_overall.csv", overall_rows)
    write_csv(out_dir / "fdr_q_sweep_by_experiment.csv", exp_rows)
    make_plots(overall_rows, out_dir)
    print(f"Wrote sweep outputs to {out_dir}")


if __name__ == "__main__":
    main()
