#!/usr/bin/env python3
"""
Submit multi-model generation/pairing/patching pipelines with Slurm dependencies.

For each model:
  1. Download model from HuggingFace into models/<model_name>
  2. For each experiment, chain:
       generate -> reject_traces -> generate_pairs -> post_process -> patch_graph
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class ModelConfig:
    local_name: str
    repo_id: str
    gen_gpus: int
    gen_time: str
    patch_time: str


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "gemma-4-31B-it": ModelConfig(
        local_name="gemma-4-31B-it",
        repo_id="google/gemma-4-31b-it",
        gen_gpus=1,
        gen_time="0-01:30:00",
        patch_time="0-12:00:00",
    ),
    "Llama-3.1-70B": ModelConfig(
        local_name="Llama-3.1-70B",
        repo_id="meta-llama/Llama-3.1-70B",
        gen_gpus=2,
        gen_time="0-02:00:00",
        patch_time="0-16:00:00",
    ),
    "gpt-oss-20b": ModelConfig(
        local_name="gpt-oss-20b",
        repo_id="openai/gpt-oss-20b",
        gen_gpus=1,
        gen_time="0-01:00:00",
        patch_time="0-10:00:00",
    ),
    "Mistral-Small-3.1-24B-Base-2503": ModelConfig(
        local_name="Mistral-Small-3.1-24B-Base-2503",
        repo_id="mistralai/Mistral-Small-3.1-24B-Base-2503",
        gen_gpus=1,
        gen_time="0-01:30:00",
        patch_time="0-12:00:00",
    ),
}


def parse_job_id(stdout: str, stderr: str) -> str:
    # Prefer stdout because `sbatch --parsable` writes the job id there, while
    # stderr can contain numeric warning text (for example memory-note messages).
    ids_stdout = re.findall(r"\b(\d{6,})\b", stdout or "")
    if ids_stdout:
        return ids_stdout[-1]

    combined = (stdout or "") + ("\n" + stderr if stderr else "")
    ids_combined = re.findall(r"\b(\d{6,})\b", combined)
    if not ids_combined:
        raise RuntimeError(f"Could not parse Slurm job id from output:\n{combined}")
    return ids_combined[-1]


def run_sbatch(args: List[str], cwd: Path) -> str:
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed (rc={proc.returncode}): {' '.join(args)}\n{combined}")
    return parse_job_id(proc.stdout or "", proc.stderr or "")


def load_experiments(auto_graph_dir: Path) -> List[Tuple[str, int]]:
    out = subprocess.check_output(
        ["python", "list_all_experiments.py"], cwd=auto_graph_dir, text=True
    )
    experiments: List[Tuple[str, int]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        exp = parts[0].strip()
        n_formats = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else 5
        experiments.append((exp, n_formats))
    return experiments


def sanitize_tag(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


def iter_model_configs(model_keys: Iterable[str]) -> Iterable[ModelConfig]:
    for key in model_keys:
        if key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key '{key}'. Allowed: {sorted(MODEL_CONFIGS)}")
        yield MODEL_CONFIGS[key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="gemma-4-31B-it,Llama-3.1-70B,gpt-oss-20b,Mistral-Small-3.1-24B-Base-2503",
        help="Comma-separated model keys.",
    )
    parser.add_argument(
        "--samples-per-format",
        type=int,
        default=10,
        help="Prompt samples per format (n_prompts = n_formats * samples_per_format).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Passed to generate.sh via MAX_NEW_TOKENS env.",
    )
    parser.add_argument(
        "--manifest",
        default="smoke_logs/multimodel_pipeline_jobs.tsv",
        help="Manifest path relative to auto_computation_graph.",
    )
    args = parser.parse_args()

    if args.samples_per_format < 1:
        raise ValueError("--samples-per-format must be >= 1")

    auto_graph_dir = Path(__file__).resolve().parent
    experiments = load_experiments(auto_graph_dir)
    model_cfgs = list(iter_model_configs([m.strip() for m in args.models.split(",") if m.strip()]))

    manifest_path = (auto_graph_dir / args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for cfg in model_cfgs:
        model_tag = sanitize_tag(cfg.local_name)

        dl_out = f"smoke_logs/dl_{model_tag}_%j.out"
        dl_job = run_sbatch(
            [
                "sbatch",
                "--parsable",
                "--job-name",
                f"dl_{model_tag}",
                "-o",
                dl_out,
                "download_model_job.sh",
                cfg.repo_id,
                cfg.local_name,
            ],
            cwd=auto_graph_dir,
        )
        print(f"[model:{cfg.local_name}] download job: {dl_job}")

        for exp, n_formats in experiments:
            n_prompts = n_formats * args.samples_per_format
            exp_tag = sanitize_tag(exp)

            gen_job = run_sbatch(
                [
                    "sbatch",
                    "--parsable",
                    "--dependency",
                    f"afterok:{dl_job}",
                    "--gpus-per-node",
                    f"h100:{cfg.gen_gpus}",
                    "--time",
                    cfg.gen_time,
                    "-o",
                    f"smoke_logs/gen_{model_tag}_{exp_tag}_%j.out",
                    "--export",
                    f"ALL,MAX_NEW_TOKENS={args.max_new_tokens}",
                    "generate.sh",
                    exp,
                    cfg.local_name,
                    str(n_prompts),
                ],
                cwd=auto_graph_dir,
            )

            reject_job = run_sbatch(
                [
                    "sbatch",
                    "--parsable",
                    "--dependency",
                    f"afterok:{gen_job}",
                    "-o",
                    f"smoke_logs/reject_{model_tag}_{exp_tag}_%j.out",
                    "reject_traces.sh",
                    exp,
                    cfg.local_name,
                ],
                cwd=auto_graph_dir,
            )

            pair_job = run_sbatch(
                [
                    "sbatch",
                    "--parsable",
                    "--dependency",
                    f"afterok:{reject_job}",
                    "-o",
                    f"smoke_logs/pair_{model_tag}_{exp_tag}_%j.out",
                    "generate_pairs.sh",
                    exp,
                    cfg.local_name,
                ],
                cwd=auto_graph_dir,
            )

            post_job = run_sbatch(
                [
                    "sbatch",
                    "--parsable",
                    "--dependency",
                    f"afterok:{pair_job}",
                    "-o",
                    f"smoke_logs/post_{model_tag}_{exp_tag}_%j.out",
                    "post_process.sh",
                    exp,
                    cfg.local_name,
                ],
                cwd=auto_graph_dir,
            )

            patch_job = run_sbatch(
                [
                    "sbatch",
                    "--parsable",
                    "--dependency",
                    f"afterok:{post_job}",
                    "--time",
                    cfg.patch_time,
                    "-o",
                    f"smoke_logs/patch_{model_tag}_{exp_tag}_%j.out",
                    "--export",
                    "ALL,MODEL_NAME={}".format(cfg.local_name),
                    "patch_graph.sh",
                    exp,
                    cfg.local_name,
                ],
                cwd=auto_graph_dir,
            )

            rows.append(
                {
                    "model_name": cfg.local_name,
                    "repo_id": cfg.repo_id,
                    "experiment": exp,
                    "n_prompts": n_prompts,
                    "download_job_id": dl_job,
                    "generate_job_id": gen_job,
                    "reject_job_id": reject_job,
                    "pair_job_id": pair_job,
                    "post_job_id": post_job,
                    "patch_job_id": patch_job,
                }
            )

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "repo_id",
                "experiment",
                "n_prompts",
                "download_job_id",
                "generate_job_id",
                "reject_job_id",
                "pair_job_id",
                "post_job_id",
                "patch_job_id",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Total experiment chains submitted: {len(rows)}")
    print(f"Total jobs submitted: {len(rows) * 5 + len(model_cfgs)}")


if __name__ == "__main__":
    main()
