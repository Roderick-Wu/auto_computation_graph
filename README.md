# auto_computation_graph

This directory implements the trace-generation and causal-patching pipeline for the hidden-variable experiments.

## Pipeline

1. Generate CoT traces with [generate_traces.py](generate_traces.py).
   - Sample questions across all prompt formats.
   - Save the raw traces to `traces.json`.

2. Validate the traces with [reject_traces.py](reject_traces.py).
   - Truncate runaway continuations so only the first trace segment is inspected.
   - Reject traces whose final answer is missing or outside tolerance.
   - Save accepted traces to `reject_traces.json`.

3. Branch into one of two patching paths.
   - Path A, counterfactual patching:
     - [intervene_generate_pairs.py](intervene_generate_pairs.py) creates counterfactual pairs.
     - [post_process_pairs.py](post_process_pairs.py) aligns token lengths.
     - [intervene_graph.py](intervene_graph.py) performs layer/token causal patching.
    - Path B, direct trace patching:
       - [intervene_graph_nopair.py](intervene_graph_nopair.py) takes `reject_traces.json` directly.
     - It does not need a counterfactual trace; it injects Gaussian noise by default.

4. Build graphs from the patching matrices with [construct_graph.py](construct_graph.py).
   - The output is a value-level causal graph built from the difference matrices.

5. Validate the causal structure with token-level intervention tests.
   - [intervene_validate_causal_structure.py](intervene_validate_causal_structure.py) tests if corrupting parent nodes changes answers (positive control) and if corrupting non-parents leaves answers unchanged (negative control).
   - Outputs: causal sensitivity metrics (hit rate, false alarm rate, specificity).
   - Corruption methods: false values, masking ([REDACTED]), counterfactual substitution.
   - Run single experiment: `bash validate_causal_structure.sh <model> <experiment>`.
   - Run all experiments: `bash validate_causal_structure_all.sh <model>`.

6. Test node necessity with node-skipping experiments.
   - [intervene_skip_nodes.py](intervene_skip_nodes.py) tests whether the model can still generate correct answers when intermediate values are truncated.
   - Progressively skips ancestors of the final answer and forces generation at different truncation points.
   - Outputs: node necessity metrics (which nodes can be bypassed, robustness score).
   - Run single experiment: `bash skip_nodes.sh <model> <experiment>`.
   - Run all experiments: `bash skip_nodes_all.sh <model>`.

## Outputs

- `traces.json` from generation
- `reject_traces.json` from validation/rejection
- `paired_traces.json` from counterfactual pairing
- `patch_runs/` from pairwise activation patching
- `patch_solo/` from direct noise patching
- `graphs/` from graph construction
- `causal_validation_results.json` from token-level intervention tests
- `node_skipping_results.json` from node necessity tests

## Notes

- The direct path is now supported by [intervene_graph_nopair.py](intervene_graph_nopair.py), which can read `reject_traces.json` directly.
- The current branch is intentionally split so counterfactual patching and noise-based patching can be compared separately.

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