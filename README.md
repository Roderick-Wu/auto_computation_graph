# auto_computation_graph

Counterfactual-patching pipeline for building value-level causal graphs from reasoning traces, plus API-based ground-truth graph generation and graph evaluation.

## Repository layout

```text
auto_computation_graph/
├── README.md
├── workspace_paths.py
├── workspace_paths.sh
├── src/        # Python entry points and prompt definitions
├── scripts/    # Slurm/local shell wrappers
└── archive/    # Legacy pipeline snapshots
```

The active code lives in `src/` and `scripts/`. Archived pre-refactor patching code lives in `archive/old_patching/`.

## Current default pipeline

1. Generate traces: `src/generate_traces.py` or `scripts/generate.sh`
2. Reject malformed or incorrect traces: `src/reject_traces.py` or `scripts/reject_traces.sh`
3. Generate counterfactual pairs: `src/intervene_generate_pairs.py` or `scripts/generate_pairs.sh`
4. Post-process and align pairs: `src/post_process_pairs.py` or `scripts/post_process.sh`
5. Run token-level counterfactual patching: `src/intervene_graph.py` or `scripts/patch_graph.sh`
6. Construct candidate graphs: `src/construct_graph.py` or `scripts/construct_graph.sh`
7. Generate API ground-truth graphs: `src/generate_ground_truth_graphs_api.py` or `scripts/generate_ground_truth_graphs_api.sh`
8. Evaluate candidate graphs: `src/evaluate_graphs_against_ground_truth.py`

## Common commands

Single experiment:

```bash
sbatch scripts/generate.sh velocity_from_ke Qwen2.5-32B 250
bash scripts/reject_traces.sh velocity_from_ke Qwen2.5-32B
bash scripts/generate_pairs.sh velocity_from_ke Qwen2.5-32B
bash scripts/post_process.sh velocity_from_ke Qwen2.5-32B
sbatch --export=ALL,MODEL_NAME=Qwen2.5-32B,PATCH_BATCH_SIZE=16 scripts/patch_graph.sh velocity_from_ke Qwen2.5-32B
bash scripts/construct_graph.sh velocity_from_ke Qwen2.5-32B pair
bash scripts/generate_ground_truth_graphs_api.sh Qwen2.5-32B velocity_from_ke
python src/evaluate_graphs_against_ground_truth.py \
  --model-name Qwen2.5-32B \
  --experiments velocity_from_ke \
  --output-dir smoke_logs/graph_eval_qwen_vs_gt_velocity_from_ke
```

All experiments:

```bash
MODEL_NAME=Qwen2.5-32B SAMPLES_PER_FORMAT=10 GPUS_PER_NODE=1 bash scripts/generate_all.sh
MODEL_NAME=Qwen2.5-32B RUN_MODE=local bash scripts/reject_traces_all.sh
MODEL_NAME=Qwen2.5-32B RUN_MODE=local bash scripts/generate_pairs_all.sh
MODEL_NAME=Qwen2.5-32B RUN_MODE=local bash scripts/post_process_all.sh
MODEL_NAME=Qwen2.5-32B PATCH_BATCH_SIZE=16 bash scripts/patch_graph_all.sh
MODEL_NAME=Qwen2.5-32B GRAPH_VARIANT=pair USE_SLURM=0 bash scripts/construct_graph_all.sh
MODEL_NAME=Qwen2.5-32B bash scripts/run_gt_qwen_all.sh
python src/evaluate_graphs_against_ground_truth.py \
  --model-name Qwen2.5-32B \
  --output-dir smoke_logs/graph_eval_qwen_vs_gt_full
```

FDR sweep:

```bash
python src/run_fdr_q_sweep.py \
  --models gpt-oss-20b,Llama3.1-70B,Mistral-Small-3.1-24B-Base-2503 \
  --fdr-q-values 0.01,0.02,0.05,0.1,0.2,0.3,0.5
```

## Output locations

By default, repo-local helpers resolve:

- `WRODERI_SCRATCH_ROOT` to `./scratch`
- `WRODERI_MODELS_ROOT` to `./models`

Per-experiment artifacts are written under:

```text
scratch/traces/<model>/<experiment>/
```

Key outputs:

- `traces.json`
- `reject_traces.json`
- `paired_traces.json`
- `aligned_pairs.json`
- `patch_runs/`
- `graphs/`
- `graphs_ground_truth_api/`

## Notes

- The pair-based path is the documented default release flow.
- Diagnostic scripts for skip-node and causal-structure validation still live in `src/` and `scripts/`, but they are not the main benchmark path.
- `scripts/generate_pairs.sh` loads `.env` from the repo root when present.
- Legacy no-pair patching code is kept in `archive/old_patching/`.

## License

MIT.
