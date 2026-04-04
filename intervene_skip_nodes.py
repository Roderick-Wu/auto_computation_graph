#!/usr/bin/env python3
"""Test node skipping: can we bypass intermediate values and still compute the answer?

This script tests the necessity of nodes by:
1. Identifying the final answer node and its parent chain
2. For each combination of parents, truncate before that parent and force-generate
3. Test progressively earlier truncations (skipping more parents)
4. Record which parent combinations are necessary vs optional

Unlike intervention_validate_causal_structure.py which corrupts parents to see if the
answer changes, this tests whether we CAN skip parents entirely and the model still
outputs the correct answer (or can compute it in a different way).

Metrics:
- Node necessity: which nodes can/cannot be skipped
- Robustness: can model recover answer even without ancestors
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import itertools

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
FINAL_ANSWER_PATTERN = re.compile(
    r"(?:the\s+answer\s+is|final\s+answer\s*(?:is|=)|answer\s*(?:is|=|:))\s*"
    r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
QUESTION_MARKER_PATTERN = re.compile(r"(?:\[\s*question\s*\]|question\s*[:\]\-])", re.IGNORECASE)


@dataclass
class SkippingTest:
    """Record of a node skipping test."""
    test_id: str
    pair_id: str
    skipped_nodes: List[str]        # nodes being skipped
    kept_nodes: List[str]           # ancestor nodes being kept
    truncation_idx: int             # where we truncate and force-generate
    baseline_answer: Optional[float]
    skipped_answer: Optional[float]
    answer_correct: bool            # did skipped version get right answer?
    answer_same: bool               # did answer match baseline?
    forced_value: str               # the value we forced the model to generate
    generation_succeeded: bool
    errors: List[str] = field(default_factory=list)


@dataclass 
class SkippingMetrics:
    """Metrics on node necessity from skipping tests."""
    total_tests: int = 0
    successful_skips: int = 0        # tests where model generated after forced value
    correct_after_skip: int = 0      # skipped tests where answer was still correct
    success_rate: float = 0.0
    robustness_score: float = 0.0    # % of successful skips with correct answer


def normalize_number_string(raw: Any) -> Optional[str]:
    """Normalize numeric string for comparison."""
    if raw is None:
        return None
    cleaned = str(raw).strip().replace(",", "")
    if not cleaned:
        return None
    try:
        dec = Decimal(cleaned)
    except Exception:
        return None
    normalized = format(dec, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"", "-0"}:
        normalized = "0"
    return normalized


def extract_final_answer_value(text: Optional[str]) -> Optional[float]:
    """Extract numeric final answer from text."""
    if not text:
        return None
    matches = list(FINAL_ANSWER_PATTERN.finditer(text))
    if matches:
        candidate = matches[-1].group(1)
        normalized = normalize_number_string(candidate)
        return float(normalized) if normalized is not None else None
    return None


def compute_relative_error(observed: float, expected: float) -> float:
    """Compute relative error."""
    denominator = max(abs(expected), 1e-12)
    return abs(observed - expected) / denominator


def load_graph_json(graph_path: Path) -> Optional[Dict]:
    """Load graph.json."""
    if not graph_path.exists():
        return None
    try:
        with open(graph_path) as f:
            return json.load(f)
    except Exception:
        return None


def load_aligned_pairs_dict(pairs_path: Path) -> Dict[str, Dict]:
    """Load aligned_pairs.json."""
    if not pairs_path.exists():
        return {}
    try:
        with open(pairs_path) as f:
            payload = json.load(f)
        pairs_dict = {}
        pairs_list = payload.get("pairs", []) if isinstance(payload, dict) else payload
        for pair in pairs_list:
            if isinstance(pair, dict):
                pair_id = str(pair.get("pair_id", pair.get("id", "")))
                if pair_id:
                    pairs_dict[pair_id] = pair
        return pairs_dict
    except Exception:
        return {}


def get_pair_trace_text(pair: Dict) -> Optional[str]:
    """Extract base trace text from pair."""
    if not isinstance(pair, dict):
        return None
    
    nested_pair = pair.get("pair", {}) if isinstance(pair.get("pair"), dict) else {}
    nested_source = nested_pair.get("source", {}) if isinstance(nested_pair.get("source"), dict) else {}
    
    trace_text = (
        nested_source.get("generated_text") or
        nested_source.get("text") or
        pair.get("source_text") or
        pair.get("source") or
        pair.get("base_text") or
        pair.get("prompt") or
        ""
    )
    return trace_text if trace_text else None


def get_pair_expected_answer(pair: Dict) -> Optional[float]:
    """Extract expected answer from pair."""
    if not isinstance(pair, dict):
        return None
    
    nested_pair = pair.get("pair", {}) if isinstance(pair.get("pair"), dict) else {}
    nested_source = nested_pair.get("source", {}) if isinstance(nested_pair.get("source"), dict) else {}
    prompt_metadata = nested_source.get("prompt_metadata", {})
    
    if not isinstance(prompt_metadata, dict):
        return None
    
    for key, value in prompt_metadata.items():
        if key.startswith("expected_"):
            normalized = normalize_number_string(value)
            if normalized is not None:
                return float(normalized)
    return None


def truncate_after_first_question(text: str, truncation_idx: int) -> str:
    """Truncate text at position and check for second question."""
    truncated = text[:truncation_idx]
    matches = list(QUESTION_MARKER_PATTERN.finditer(text))
    if len(matches) > 1 and matches[1].start() >= truncation_idx:
        return truncated
    return truncated


def generate_from_truncation(
    model: Any,
    tokenizer: Any,
    truncated_text: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
) -> Tuple[str, int]:
    """Generate continuation from truncated text."""
    try:
        encoded = tokenizer(truncated_text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        num_new = outputs[0].shape[0] - input_ids.shape[1]
        
        return full_text, int(num_new)
    except Exception:
        return truncated_text, 0


def generate_batch(
    model: Any,
    tokenizer: Any,
    truncated_texts: List[str],
    max_new_tokens: int = 100,
    temperature: float = 0.7,
) -> List[Tuple[str, int]]:
    """Generate continuations for a batch of truncated texts in parallel."""
    if not truncated_texts:
        return []
    
    if len(truncated_texts) == 1:
        result = generate_from_truncation(model, tokenizer, truncated_texts[0], max_new_tokens, temperature)
        return [result]
    
    try:
        # Tokenize batch with padding
        encoded = tokenizer(
            truncated_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        
        # Generate in batch
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode results
        results = []
        for i, output_ids in enumerate(outputs):
            full_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            input_len = input_ids[i].shape[0]
            num_new = output_ids.shape[0] - input_len
            results.append((full_text, int(num_new)))
        
        return results
    except Exception:
        # Fallback to sequential generation
        return [generate_from_truncation(model, tokenizer, text, max_new_tokens, temperature) for text in truncated_texts]


def build_parent_chain(node_id: str, graph: Dict) -> List[str]:
    """Get all ancestors of a node in topological order (child to root)."""
    edges = graph.get("edges", [])
    parents_of = defaultdict(set)
    for edge in edges:
        src, dst = edge["source"], edge["target"]
        parents_of[dst].add(src)
    
    chain = [node_id]
    to_visit = list(parents_of.get(node_id, set()))
    
    while to_visit:
        current = to_visit.pop(0)
        if current not in chain:
            chain.append(current)
            to_visit.extend(parents_of.get(current, set()))
    
    return chain


def test_node_skipping_on_pair(
    model: Any,
    tokenizer: Any,
    pair_id: str,
    pair: Dict,
    graph: Dict,
    max_skip_depths: int = 3,
    max_tests: int = 20,
    batch_size: int = 8,
) -> List[SkippingTest]:
    """Test skipping nodes for a single pair with batched generation."""
    
    results = []
    trace_text = get_pair_trace_text(pair)
    expected_answer = get_pair_expected_answer(pair)
    
    if not trace_text or expected_answer is None:
        return results
    
    # Get baseline answer
    baseline_answer = extract_final_answer_value(trace_text)
    if baseline_answer is None:
        return results
    
    node_stats = graph.get("node_stats", {})
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    
    # Focus on final answer (v0) and intermediate values
    reachable_nodes = [n for n in nodes.keys() if n.startswith("v")]
    
    # Batch up all skip tests to run in parallel
    pending_tests: List[Tuple[SkippingTest, str, str]] = []  # (test_obj, forced_prompt, target_node)
    
    for target_node in reachable_nodes[:3]:  # Test a few key nodes
        if len(results) + len(pending_tests) >= max_tests:
            break
            
        parent_chain = build_parent_chain(target_node, graph)
        
        # Try different depths of skipping
        for skip_depth in range(1, min(max_skip_depths + 1, len(parent_chain))):
            if len(results) + len(pending_tests) >= max_tests:
                break
            
            # Nodes to skip: all except the kept ones
            nodes_to_keep = parent_chain[:skip_depth]
            nodes_to_skip = [n for n in parent_chain[skip_depth:] if n != target_node]
            
            if not nodes_to_skip:
                continue
            
            # Find truncation point: before earliest skipped node
            earliest_skip_node = nodes_to_skip[0]
            trunc_idx = node_stats.get(earliest_skip_node, {}).get("truncation_token_index", -1)
            
            if trunc_idx < 0:
                continue
            
            # Get the "forced value" - what we expect the model to generate
            target_values = nodes[target_node].get("value_texts", [])
            if not target_values:
                continue
            
            forced_value = target_values[0]
            forced_prefix = f"{forced_value} = "
            
            # Build truncation point
            truncated = truncate_after_first_question(trace_text, trunc_idx)
            if not truncated or truncated.endswith(" ="):
                truncated = truncated.rstrip("= ").rstrip()
            
            # Append forced value generation prompt
            forced_prompt = truncated + " ... " + forced_prefix
            
            test = SkippingTest(
                test_id=f"{pair_id}_{target_node}_skip{skip_depth}",
                pair_id=pair_id,
                skipped_nodes=nodes_to_skip,
                kept_nodes=nodes_to_keep,
                truncation_idx=trunc_idx,
                baseline_answer=baseline_answer,
                skipped_answer=None,
                answer_correct=False,
                answer_same=False,
                forced_value=forced_value,
                generation_succeeded=False,
            )
            pending_tests.append((test, forced_prompt, target_node))
    
    # Process pending tests in batches
    for batch_start in range(0, len(pending_tests), batch_size):
        batch_end = min(batch_start + batch_size, len(pending_tests))
        batch = pending_tests[batch_start:batch_end]
        
        # Generate all in batch
        forced_prompts = [prompt for (_, prompt, _) in batch]
        generations = generate_batch(model, tokenizer, forced_prompts, max_new_tokens=100, temperature=0.7)
        
        # Update results
        for i, (test, forced_prompt, target_node) in enumerate(batch):
            generated, gen_len = generations[i]
            skipped_answer = extract_final_answer_value(generated)
            
            # Check correctness
            answer_correct = False
            if skipped_answer is not None:
                error = compute_relative_error(skipped_answer, expected_answer)
                answer_correct = error < 0.05
            
            answer_same = (
                skipped_answer is not None and baseline_answer is not None and
                abs(skipped_answer - baseline_answer) < 1e-9
            )
            
            test.skipped_answer = skipped_answer
            test.answer_correct = answer_correct
            test.answer_same = answer_same
            test.generation_succeeded = gen_len > 0
            
            results.append(test)
    
    return results


def aggregate_skipping_metrics(results: List[SkippingTest]) -> SkippingMetrics:
    """Compute metrics from skipping tests."""
    metrics = SkippingMetrics(total_tests=len(results))
    
    if not results:
        return metrics
    
    successful = [r for r in results if r.generation_succeeded]
    metrics.successful_skips = len(successful)
    
    if successful:
        metrics.success_rate = len(successful) / len(results)
        correct = sum(1 for r in successful if r.answer_correct)
        metrics.correct_after_skip = correct
        metrics.robustness_score = correct / len(successful) if successful else 0.0
    
    return metrics


def resolve_default_model_path(model_name: str) -> Path:
    return REPO_ROOT / "models" / model_name


def resolve_default_graph_dir(model_name: str, experiment: str) -> Path:
    scratch_root = Path.home() / "links" / "scratch"
    if not scratch_root.exists():
        scratch_root = Path.home() / "scratch"
    return scratch_root / "traces" / model_name / experiment / "patch_runs"


def resolve_default_output_json(model_name: str, experiment: str) -> Path:
    scratch_root = Path.home() / "links" / "scratch"
    if not scratch_root.exists():
        scratch_root = Path.home() / "scratch"
    return scratch_root / "traces" / model_name / experiment / "node_skipping_results.json"


def main():
    parser = argparse.ArgumentParser(
        description="Test node skipping: can we bypass intermediate values?"
    )
    parser.add_argument("--model-name", type=str, default="Qwen2.5-72B", help="Model folder name.")
    parser.add_argument("--experiment", type=str, default="velocity", help="Experiment name.")
    parser.add_argument("--graph-dir", type=Path, default=None, help="Directory with graphs.")
    parser.add_argument("--output-json", type=Path, default=None, help="Output results.")
    parser.add_argument("--max-pairs", type=int, default=5, help="Max pairs to test.")
    parser.add_argument("--max-skip-depths", type=int, default=3, help="Max depth of skips.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for generation.")
    parser.add_argument("--device", type=str, default="cuda", help="Device.")
    parser.add_argument("--dtype", type=str, default="float16", help="Model dtype.")
    
    args = parser.parse_args()
    
    model_path = resolve_default_model_path(args.model_name)
    graph_dir = args.graph_dir or resolve_default_graph_dir(args.model_name, args.experiment)
    output_json = args.output_json or resolve_default_output_json(args.model_name, args.experiment)
    
    if not graph_dir.exists():
        raise FileNotFoundError(f"Graph directory not found: {graph_dir}")
    
    print(f"Loading model from: {model_path}")
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=dtype, device_map=args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    print(f"Loading graphs from: {graph_dir}")
    
    pair_dirs = sorted([d for d in graph_dir.iterdir() if d.is_dir()])[:args.max_pairs]
    pairs_json = graph_dir.parent / "aligned_pairs.json"
    aligned_pairs_dict = load_aligned_pairs_dict(pairs_json)
    
    all_results: List[SkippingTest] = []
    
    for pair_dir in pair_dirs:
        pair_name = pair_dir.name
        graph_path = pair_dir / "graph.json"
        
        if not graph_path.exists():
            print(f"Skipping {pair_name}: no graph.json")
            continue
        
        graph = load_graph_json(graph_path)
        pair = aligned_pairs_dict.get(pair_name)
        
        if not graph or not pair:
            continue
        
        print(f"Testing skipping for {pair_name}...")
        
        results = test_node_skipping_on_pair(
            model, tokenizer,
            pair_name, pair, graph,
            max_skip_depths=args.max_skip_depths,
            max_tests=20,
            batch_size=args.batch_size,
        )
        all_results.extend(results)
        print(f"  {len(results)} skipping tests completed")
    
    metrics = aggregate_skipping_metrics(all_results)
    
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    output_payload = {
        "experiment": args.experiment,
        "model": args.model_name,
        "metrics": asdict(metrics),
        "results": [asdict(r) for r in all_results],
    }
    
    with open(output_json, "w") as f:
        json.dump(output_payload, f, indent=2)
    
    print(f"\nResults saved to: {output_json}")
    print(f"Total skip tests: {metrics.total_tests}")
    print(f"Successful skips: {metrics.successful_skips}/{metrics.total_tests} ({metrics.success_rate:.1%})")
    print(f"Robustness score: {metrics.robustness_score:.1%}")


if __name__ == "__main__":
    main()
