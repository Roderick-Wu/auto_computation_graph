#!/usr/bin/env python3
"""Analyze acceptance rates across experiments.

Reads traces.json and reject_traces.json for each experiment to compute:
- Per-experiment acceptance rate (% of traces that passed validation)
- Total traces generated vs accepted
- Rejection reasons breakdown
- Identifies experiments with low acceptance (bottlenecks)
"""

import argparse
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RejectionStats:
    """Statistics for a single experiment's rejection."""
    experiment: str
    total_traces: int = 0
    accepted_traces: int = 0
    acceptance_rate: float = 0.0
    rejected_count: int = 0
    rejection_rate: float = 0.0
    
    # Breakdown by rejection reason
    missing_expected_answer: int = 0
    missing_final_answer: int = 0
    outside_tolerance: int = 0
    empty_trace: int = 0
    
    errors: List[str] = field(default_factory=list)


@dataclass
class RejectionReport:
    """Aggregate rejection statistics across all experiments."""
    model_name: str
    total_experiments: int = 0
    total_traces_generated: int = 0
    total_traces_accepted: int = 0
    overall_acceptance_rate: float = 0.0
    
    # Experiments flagged for concern
    low_acceptance_experiments: List[str] = field(default_factory=list)  # < 50% acceptance
    concerning_experiments: List[str] = field(default_factory=list)       # 50-70% acceptance
    
    per_experiment_stats: Dict[str, RejectionStats] = field(default_factory=dict)
    
    # Overall rejection breakdown
    total_missing_expected: int = 0
    total_missing_final: int = 0
    total_outside_tolerance: int = 0
    total_empty_traces: int = 0


def resolve_default_model_path(model_name: str) -> Path:
    return REPO_ROOT / "models" / model_name


def resolve_traces_dir(model_name: str, experiment: str) -> Path:
    """Resolve path to experiment traces directory."""
    scratch_root = Path.home() / "scratch"
    return scratch_root / "traces" / model_name / experiment


def get_all_experiments(model_name: str) -> List[str]:
    """Get list of all registered experiments from list_all_experiments.py."""
    try:
        result = subprocess.run(
            ["python", "list_all_experiments.py"],
            cwd=str(REPO_ROOT / "auto_computation_graph"),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        
        experiments = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split()
                if parts:
                    experiments.append(parts[0])
        return experiments
    except Exception as e:
        print(f"Warning: Could not get experiments list: {e}")
        return []


def load_trace_count(traces_path: Path) -> int:
    """Load count of traces from traces.json."""
    if not traces_path.exists():
        return 0
    try:
        with open(traces_path) as f:
            payload = json.load(f)
        
        if isinstance(payload, list):
            return len(payload)
        if isinstance(payload, dict) and isinstance(payload.get("traces"), list):
            return len(payload["traces"])
        return 0
    except Exception:
        return 0


def load_reject_traces_stats(reject_traces_path: Path) -> tuple[int, Dict[str, int]]:
    """Load reject_traces.json and extract acceptance count and rejection reasons.
    
    Returns (accepted_count, rejection_reason_counts).
    """
    if not reject_traces_path.exists():
        return 0, {}
    
    try:
        with open(reject_traces_path) as f:
            payload = json.load(f)
        
        traces = []
        if isinstance(payload, list):
            traces = payload
        elif isinstance(payload, dict) and isinstance(payload.get("traces"), list):
            traces = payload["traces"]
        
        # Count acceptance
        accepted_count = len(traces)
        
        # Try to extract rejection metadata if available
        rejection_counts = defaultdict(int)
        metadata = payload.get("rejection_metadata", {}) if isinstance(payload, dict) else {}
        
        if isinstance(metadata, dict):
            for key, count in metadata.items():
                rejection_counts[key] = count
        
        return accepted_count, dict(rejection_counts)
    except Exception:
        return 0, {}


def analyze_experiment(model_name: str, experiment: str) -> RejectionStats:
    """Analyze rejection stats for a single experiment."""
    stats = RejectionStats(experiment=experiment)
    
    traces_dir = resolve_traces_dir(model_name, experiment)
    traces_path = traces_dir / "traces.json"
    reject_traces_path = traces_dir / "reject_traces.json"
    
    # Load total traces
    total = load_trace_count(traces_path)
    stats.total_traces = total
    
    # Load accepted traces and rejection breakdown
    accepted, rejection_breakdown = load_reject_traces_stats(reject_traces_path)
    stats.accepted_traces = accepted
    
    if total > 0:
        stats.rejection_rate = (total - accepted) / total
        stats.acceptance_rate = accepted / total
    
    # Extract rejection reasons
    stats.missing_expected_answer = rejection_breakdown.get("missing_expected_answer", 0)
    stats.missing_final_answer = rejection_breakdown.get("missing_final_answer", 0)
    stats.outside_tolerance = rejection_breakdown.get("outside_tolerance", 0)
    stats.empty_trace = rejection_breakdown.get("empty_trace", 0)
    stats.rejected_count = total - accepted
    
    if stats.total_traces == 0:
        stats.errors.append(f"No traces.json found at {traces_path}")
    
    if stats.accepted_traces == 0 and stats.total_traces > 0:
        stats.errors.append(f"No accepted traces found (0/{stats.total_traces} acceptance)")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze acceptance rates across all experiments."
    )
    parser.add_argument("--model-name", type=str, default="Qwen2.5-72B", help="Model folder name.")
    parser.add_argument("--output-json", type=Path, default=None, help="Output report JSON.")
    parser.add_argument("--low-threshold", type=float, default=0.50, help="Threshold for 'low acceptance' warning (default 50%).")
    parser.add_argument("--concerning-threshold", type=float, default=0.70, help="Threshold for 'concerning acceptance' warning (default 70%).")
    
    args = parser.parse_args()
    
    print(f"Analyzing rejection rates for: {args.model_name}")
    
    # Get all experiments
    experiments = get_all_experiments(args.model_name)
    if not experiments:
        print("Warning: Could not discover experiments. Check list_all_experiments.py")
        experiments = []
    
    print(f"Found {len(experiments)} experiments")
    
    # Analyze each experiment
    report = RejectionReport(model_name=args.model_name, total_experiments=len(experiments))
    
    for experiment in experiments:
        print(f"  {experiment}...", end=" ", flush=True)
        stats = analyze_experiment(args.model_name, experiment)
        report.per_experiment_stats[experiment] = stats
        
        report.total_traces_generated += stats.total_traces
        report.total_traces_accepted += stats.accepted_traces
        
        # Track rejection reasons
        report.total_missing_expected += stats.missing_expected_answer
        report.total_missing_final += stats.missing_final_answer
        report.total_outside_tolerance += stats.outside_tolerance
        report.total_empty_traces += stats.empty_trace
        
        # Flag concerning experiments
        if stats.acceptance_rate < args.low_threshold:
            report.low_acceptance_experiments.append(experiment)
            print(f"[LOW {stats.acceptance_rate:.1%}]")
        elif stats.acceptance_rate < args.concerning_threshold:
            report.concerning_experiments.append(experiment)
            print(f"[CONCERNING {stats.acceptance_rate:.1%}]")
        else:
            print(f"[OK {stats.acceptance_rate:.1%}]")
    
    # Compute overall acceptance rate
    if report.total_traces_generated > 0:
        report.overall_acceptance_rate = report.total_traces_accepted / report.total_traces_generated
    
    # Save report
    if args.output_json is None:
        scratch_root = Path.home() / "scratch"
        args.output_json = scratch_root / "traces" / args.model_name / "rejection_analysis.json"
    
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    
    output_payload = {
        "model": args.model_name,
        "overall_stats": {
            "total_experiments": report.total_experiments,
            "total_traces_generated": report.total_traces_generated,
            "total_traces_accepted": report.total_traces_accepted,
            "overall_acceptance_rate": report.overall_acceptance_rate,
            "low_acceptance_count": len(report.low_acceptance_experiments),
            "concerning_acceptance_count": len(report.concerning_experiments),
        },
        "rejection_breakdown": {
            "missing_expected_answer": report.total_missing_expected,
            "missing_final_answer": report.total_missing_final,
            "outside_tolerance": report.total_outside_tolerance,
            "empty_trace": report.total_empty_traces,
        },
        "low_acceptance_experiments": report.low_acceptance_experiments,
        "concerning_experiments": report.concerning_experiments,
        "per_experiment": {
            exp: asdict(stats) for exp, stats in report.per_experiment_stats.items()
        },
    }
    
    with open(args.output_json, "w") as f:
        json.dump(output_payload, f, indent=2)
    
    print(f"\nReport saved to: {args.output_json}")
    print(f"\n{'='*60}")
    print(f"Summary for {args.model_name}")
    print(f"{'='*60}")
    print(f"Total experiments: {report.total_experiments}")
    print(f"Total traces generated: {report.total_traces_generated:,}")
    print(f"Total traces accepted: {report.total_traces_accepted:,}")
    print(f"Overall acceptance rate: {report.overall_acceptance_rate:.2%}")
    print()
    
    if report.low_acceptance_experiments:
        print(f"⚠️  LOW ACCEPTANCE (< {args.low_threshold:.0%}): {len(report.low_acceptance_experiments)}")
        for exp in sorted(report.low_acceptance_experiments):
            stats = report.per_experiment_stats[exp]
            print(f"   {exp}: {stats.acceptance_rate:.1%} ({stats.accepted_traces}/{stats.total_traces})")
        print()
    
    if report.concerning_experiments:
        print(f"⚠️  CONCERNING ACCEPTANCE ({args.low_threshold:.0%}-{args.concerning_threshold:.0%}): {len(report.concerning_experiments)}")
        for exp in sorted(report.concerning_experiments):
            stats = report.per_experiment_stats[exp]
            print(f"   {exp}: {stats.acceptance_rate:.1%} ({stats.accepted_traces}/{stats.total_traces})")
        print()
    
    # Rejection reason breakdown
    print(f"Rejection reason breakdown:")
    print(f"  Missing expected answer: {report.total_missing_expected:,}")
    print(f"  Missing final answer: {report.total_missing_final:,}")
    print(f"  Outside tolerance: {report.total_outside_tolerance:,}")
    print(f"  Empty trace: {report.total_empty_traces:,}")
    print()


if __name__ == "__main__":
    main()
