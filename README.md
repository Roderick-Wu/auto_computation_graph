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

## Outputs

- `traces.json` from generation
- `reject_traces.json` from validation/rejection
- `paired_traces.json` from counterfactual pairing
- `patch_runs/` from pairwise activation patching
- `patch_solo/` from direct noise patching
- `graphs/` from graph construction

## Notes

- The direct path is now supported by [intervene_graph_nopair.py](intervene_graph_nopair.py), which can read `reject_traces.json` directly.
- The current branch is intentionally split so counterfactual patching and noise-based patching can be compared separately.
- A later follow-up script should validate the constructed graphs with token-level patching.