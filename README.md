# auto_computation_graph

This directory implements the trace-generation and causal-patching pipeline for the hidden-variable experiments.

## Pipeline

1. Generate CoT traces with [generate_traces.py](generate_traces.py).
   - Sample questions across all prompt formats.
   - Save the raw traces to `traces.json`.

2. Validate the traces with [reject_traces.py](reject_traces.py).
   - Truncate runaway continuations so only the first trace segment is inspected.
   - Reject traces whose final answer is missing or outside tolerance.
   - Save the accepted traces to `reject_traces.json`.

3. Run counterfactual patching.
   - [intervene_generate_pairs.py](intervene_generate_pairs.py) creates counterfactual pairs.
   - [post_process_pairs.py](post_process_pairs.py) aligns token lengths.
   - [intervene_graph.py](intervene_graph.py) performs token-level causal patching.

4. Build graphs from the patching matrices with [construct_graph.py](construct_graph.py).
   - The output is a value-level causal graph built from the difference matrices.

5. Build API-based ground-truth graphs.
   - [generate_ground_truth_graphs_api.py](generate_ground_truth_graphs_api.py) creates a value graph from prompt + labeled CoT via API calls.
   - Wrapper for a single experiment: `bash generate_ground_truth_graphs_api.sh <model> <experiment>`.
   - Wrapper for all Qwen experiments: `bash run_gt_qwen_all.sh`.
   - Ground-truth generation must run on a login node (compute nodes do not have outbound API access).

6. Evaluate candidate patching graphs against API ground truth.
   - [evaluate_graphs_against_ground_truth.py](evaluate_graphs_against_ground_truth.py) compares `graphs/` against `graphs_ground_truth_api/`.
   - Reports direct-edge and transitive-closure metrics per pair, per experiment, and overall.

7. Validate the causal structure with token-level intervention tests.
   - [intervene_validate_causal_structure.py](intervene_validate_causal_structure.py) tests if corrupting parent nodes changes answers (positive control) and if corrupting non-parents leaves answers unchanged (negative control).
   - Outputs: causal sensitivity metrics (hit rate, false alarm rate, specificity).
   - Corruption methods: false values, masking ([REDACTED]), counterfactual substitution.
   - Run single experiment: `bash validate_causal_structure.sh <model> <experiment>`.
   - Run all experiments: `bash validate_causal_structure_all.sh <model>`.

8. Test node necessity with node-skipping experiments.
   - [intervene_skip_nodes.py](intervene_skip_nodes.py) tests whether the model can still generate correct answers when intermediate values are truncated.
   - Progressively skips ancestors of the final answer and forces generation at different truncation points.
   - Outputs: node necessity metrics (which nodes can be bypassed, robustness score).
   - Run single experiment: `bash skip_nodes.sh <model> <experiment>`.
   - Run all experiments: `bash skip_nodes_all.sh <model>`.

## Outputs

- `traces.json` from generation
- `reject_traces.json` from validation/rejection (accepted traces)
- `paired_traces.json` from counterfactual pairing
- `patch_runs/` from pairwise activation patching
- `graphs/` from graph construction
- `graphs_ground_truth_api/` from API-based ground-truth graph generation
- `graph_eval_*/` (under `smoke_logs/`) from candidate-vs-ground-truth evaluation
- `causal_validation_results.json` from token-level intervention tests
- `node_skipping_results.json` from node necessity tests

## Notes

- The nopair/noise patching branch is intentionally omitted from this README because it is still work in progress.
- Gemma-4-31B requires a recent Transformers build with `gemma4` support.
- `generate_traces.py` includes Gemma tokenizer fallback logic (`extra_special_tokens={}`) to avoid tokenizer config incompatibilities on some installations.

## Ground-Truth Graphs

### Generate Ground Truth (API)

Single experiment:
```bash
bash generate_ground_truth_graphs_api.sh Qwen2.5-32B velocity_from_ke
```

All experiments:
```bash
bash run_gt_qwen_all.sh
```

Outputs are written to:
`/scratch/<user>/traces/<model>/<experiment>/graphs_ground_truth_api/`

### Evaluate Candidate Graphs vs Ground Truth

Run evaluation:
```bash
python evaluate_graphs_against_ground_truth.py \
  --model-name Qwen2.5-32B \
  --output-dir smoke_logs/graph_eval_qwen_vs_gt_full
```

Generated files:
- `overall_metrics.tsv`
- `experiment_metrics.tsv`
- `pair_metrics.tsv`
- `summary.json`

Current Qwen2.5-32B full-run snapshot:
- Edge micro: `P=0.365940`, `R=0.666059`, `F1=0.472360`, `J=0.309209`
- Closure micro: `P=0.781966`, `R=0.727767`, `F1=0.753894`, `J=0.605000`

## Graph Validation

The pipeline includes two types of causal validation to verify the constructed graphs reflect true computational structure:

### Token-Level Intervention (`intervene_validate_causal_structure.py`)

Tests the causal assumptions by corrupting nodes and observing whether answers change:

- **Positive Control**: Corrupt direct parents of a value → answer should change (hit rate).
- **Negative Control**: Corrupt non-parent nodes → answer should remain unchanged (false alarm rate).
- **Corruption Methods**:
  - False values: replace numbers with incorrect ones (e.g., 0.57 → 1.14)
  - Masking: replace with [REDACTED] or $VALUE placeholders
  - Counterfactuals: use values from paired traces when available
  
- **Metrics**:
  - **Sensitivity**: % of positive controls where answer changed (target: high)
  - **Specificity**: % of negative controls where answer remained unchanged (target: high)
  - **Hit-rate by method**: how effective each corruption strategy is

Example output: "Sensitivity: 85%, Specificity: 92%" means the graph correctly identifies 85% of true parents and has only 8% false edges.

### Node Skipping (`intervene_skip_nodes.py`)

Tests whether intermediate values are necessary by progressively truncating them:

1. Select the final answer value and its parent chain
2. For different truncation depths:
   - Truncate before a parent node
   - Force the model to continue generating: `... [value] = `
   - Check if the model can still compute the correct answer
3. Record which ancestors can be bypassed

- **Metrics**:
  - **Success rate**: % of skip attempts where model generated text
  - **Robustness score**: % of successful skips with correct answers (target: should decrease as more ancestors are skipped)

Example: If the final answer is `0.57 = 30 / 52.8`, we test:
- Can model compute this if we hide the `52.8`? (one parent truncated)
- Can model compute this if we hide both `30` and `52.8`? (all parents truncated)
- Higher robustness score indicates some parents are redundant or reconstructible

## Usage Examples

Single experiment on single pair:
```bash
bash validate_causal_structure.sh Qwen2.5-72B velocity
bash skip_nodes.sh Qwen2.5-72B velocity
```

All experiments on a model (uses registered experiments from `prompts.py`):
```bash
bash validate_causal_structure_all.sh Qwen2.5-72B
bash skip_nodes_all.sh Qwen2.5-72B
```

Custom graph directory:
```bash
python intervene_validate_causal_structure.py --model-name Qwen2.5-72B --graph-dir /path/to/patch_runs
```
