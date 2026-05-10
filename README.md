# auto_computation_graph

This directory contains the full counterfactual-patching pipeline for building value-level causal graphs from reasoning traces, plus API-based "ground-truth" graph generation and quantitative graph evaluation.

## Current default pipeline (pair-only)

The active flow is the **counterfactual pair** variant:

1. Generate traces (`traces.json`)  
   Script: `generate_traces.py` (wrapper: `generate.sh`, sweep: `generate_all.sh`)
2. Reject malformed/incorrect traces (`reject_traces.json`)  
   Script: `reject_traces.py` (wrapper: `reject_traces.sh`, sweep: `reject_traces_all.sh`)
3. Generate counterfactual pairs (`paired_traces.json`)  
   Script: `intervene_generate_pairs.py` (wrapper: `generate_pairs.sh`, sweep: `generate_pairs_all.sh`)
4. Post-process / token-align pairs (`aligned_pairs.json`)  
   Script: `post_process_pairs.py` (wrapper: `post_process.sh`, sweep: `post_process_all.sh`)
5. Run token-level counterfactual patching (`patch_runs/`)  
   Script: `intervene_graph.py` (wrapper: `patch_graph.sh`, sweep: `patch_graph_all.sh`)
6. Construct candidate causal graphs (`graphs/`)  
   Script: `construct_graph.py` (wrapper: `construct_graph.sh`, sweep: `construct_graph_all.sh`)
7. Generate API-based ground-truth graphs (`graphs_ground_truth_api/`)  
   Script: `generate_ground_truth_graphs_api.py` (wrappers: `generate_ground_truth_graphs_api.sh`, `run_gt_qwen_all.sh`)
8. Evaluate candidate graphs vs ground-truth graphs  
   Script: `evaluate_graphs_against_ground_truth.py`

## Where each stage should run

- **GPU Slurm job**:
  - `generate.sh`
  - `patch_graph.sh`
- **CPU/local (login node preferred)**:
  - `reject_traces.sh` / `reject_traces_all.sh` (can be local or Slurm)
  - `post_process.sh` / `post_process_all.sh`
  - `construct_graph.sh` / `construct_graph_all.sh`
  - `evaluate_graphs_against_ground_truth.py`
- **Login node only (needs outbound internet/API)**:
  - `generate_pairs.sh` / `generate_pairs_all.sh`
  - `generate_ground_truth_graphs_api.py` / `generate_ground_truth_graphs_api.sh` / `run_gt_qwen_all.sh`

## Graph Construction

The paired-counterfactual graph constructor defaults to a BH-only rule:

```bash
GRAPH_VARIANT=pair GRAPH_RENDER=none MODEL_NAME=Qwen2.5-32B bash construct_graph_all.sh
```

Default paired settings:
- `SELECTION_METHOD=fdr`
- `FDR_Q=0.2`
- `PARENT_CAUSAL_RULE=bh_only`
- `MIN_TOKENS=0`
- `EDGE_BUILD_SCOPE=all_nodes`

In this mode, [construct_graph.py](construct_graph.py) first applies BH FDR control to the layer-aggregated patching scores for each child value.
It then keeps a parent edge for every earlier value whose own token span contains at least one BH-selected token.
Selected outlier tokens outside value spans are ignored.
No relative threshold or fallback parent rule is applied.

To reproduce the older relative-pruning rule:

```bash
PARENT_CAUSAL_RULE=strongest_plus_relative RELATIVE_EDGE_THRESHOLD=0.3 MIN_TOKENS=1 \
  GRAPH_VARIANT=pair GRAPH_RENDER=none MODEL_NAME=Qwen2.5-32B bash construct_graph_all.sh
```

To sweep the BH FDR threshold and produce paper-friendly CSVs/plots:

```bash
python run_fdr_q_sweep.py \
  --models gpt-oss-20b,Llama3.1-70B,Mistral-Small-3.1-24B-Base-2503 \
  --fdr-q-values 0.01,0.02,0.05,0.1,0.2,0.3,0.5
```

The sweep writes separate candidate graph directories named
`graphs_bh_only_fdr_q_<q>` under each trace experiment, evaluates each setting
against `graphs_ground_truth_api`, and writes summaries to
`compiled_results/fdr_q_sweep/`.

## Typical commands

Single experiment:

```bash
# 1) traces
sbatch generate.sh velocity_from_ke Qwen2.5-32B 250

# 2) reject
bash reject_traces.sh velocity_from_ke Qwen2.5-32B

# 3) pair generation (API)
bash generate_pairs.sh velocity_from_ke Qwen2.5-32B

# 4) post-process
bash post_process.sh velocity_from_ke Qwen2.5-32B

# 5) patching
sbatch --export=ALL,MODEL_NAME=Qwen2.5-32B,PATCH_BATCH_SIZE=16 patch_graph.sh velocity_from_ke Qwen2.5-32B

# 6) graph construction (fast CPU parse)
bash construct_graph.sh velocity_from_ke Qwen2.5-32B pair

# 7) API ground-truth graph
bash generate_ground_truth_graphs_api.sh Qwen2.5-32B velocity_from_ke

# 8) evaluate candidate vs GT
python evaluate_graphs_against_ground_truth.py \
  --model-name Qwen2.5-32B \
  --experiments velocity_from_ke \
  --output-dir smoke_logs/graph_eval_qwen_vs_gt_velocity_from_ke
```

All experiments:

```bash
# GPU traces
MODEL_NAME=Qwen2.5-32B SAMPLES_PER_FORMAT=10 GPUS_PER_NODE=1 bash generate_all.sh

# CPU/API pipeline
MODEL_NAME=Qwen2.5-32B RUN_MODE=local bash reject_traces_all.sh
MODEL_NAME=Qwen2.5-32B RUN_MODE=local bash generate_pairs_all.sh
MODEL_NAME=Qwen2.5-32B RUN_MODE=local bash post_process_all.sh

# GPU patching
MODEL_NAME=Qwen2.5-32B PATCH_BATCH_SIZE=16 bash patch_graph_all.sh

# CPU graph construction
MODEL_NAME=Qwen2.5-32B GRAPH_VARIANT=pair USE_SLURM=0 bash construct_graph_all.sh

# API ground-truth (login node)
MODEL_NAME=Qwen2.5-32B bash run_gt_qwen_all.sh

# Full evaluation
python evaluate_graphs_against_ground_truth.py \
  --model-name Qwen2.5-32B \
  --output-dir smoke_logs/graph_eval_qwen_vs_gt_full
```

## Output files

Per experiment under:
`/scratch/<user>/traces/<model>/<experiment>/`

Key artifacts:

- `traces.json`
- `reject_traces.json`
- `paired_traces.json`
- `aligned_pairs.json`
- `patch_runs/` and `patch_runs/patching_summary.json`
- `graphs/` and `graphs/pair*/graph.json`
- `graphs_ground_truth_api/` and `graphs_ground_truth_api/graph_summary.json`

Evaluation artifacts:

- `smoke_logs/graph_eval_*/overall_metrics.tsv`
- `smoke_logs/graph_eval_*/experiment_metrics.tsv`
- `smoke_logs/graph_eval_*/pair_metrics.tsv`
- `smoke_logs/graph_eval_*/summary.json`

## Current full-run snapshots

Qwen2.5-32B (`smoke_logs/graph_eval_qwen_vs_gt_full/overall_metrics.tsv`):

- Edge micro: `P=0.365940`, `R=0.666059`, `F1=0.472360`, `J=0.309209`
- Closure micro: `P=0.781966`, `R=0.727767`, `F1=0.753894`, `J=0.605000`
- Pairs compared: `1254`

Gemma-4-31B (`smoke_logs/graph_eval_gemma_vs_gt_full_20260506_155555_rerun/overall_metrics.tsv`):

- Edge micro: `P=0.338216`, `R=0.913570`, `F1=0.493669`, `J=0.327729`
- Closure micro: `P=0.756385`, `R=0.938316`, `F1=0.837585`, `J=0.720556`
- Pairs compared: `1026`

## Important implementation notes

- The nopair/noise branch is intentionally excluded from the main documented flow.
- `generate_pairs.sh` now auto-loads `.env` (if present) so API keys/config are available during pair generation.
- `intervene_generate_pairs.py` supports environments without `python-dotenv` installed (safe fallback import).
- `intervene_graph.py` now includes CUDA OOM backoff during patch batching (automatic batch-size halving retry).
- `intervene_graph.py` writes model metadata from the actual model path instead of hardcoded model names.
- Gemma-4-31B requires a recent Transformers build with `gemma4` support.

## Legacy/diagnostic scripts

These scripts still exist, but are no longer the primary graph-quality benchmark:

- `intervene_validate_causal_structure.py`
- `intervene_skip_nodes.py`

They remain useful for targeted debugging, while the primary benchmark is now candidate-vs-ground-truth graph comparison.
