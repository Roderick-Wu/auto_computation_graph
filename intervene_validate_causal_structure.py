#!/usr/bin/env python3
"""Validate causal structure by intervening at token level.

This script tests the causal assumptions in the graph by:
1. Positive control: Corrupt direct parents of a value -> answer should change
2. Negative control: Corrupt non-parents -> answer should remain unchanged

Corruption methods:
- False values: replace numeric values with incorrect ones
- Masking: replace with [REDACTED] or other placeholders
- Counterfactuals: use values from counterfactual pairs when available

Metrics aggregated across all tested values:
- Hit rate: % of positive controls where answer changed (higher is better)
- False alarm rate: % of negative controls where answer changed (lower is better)
- Specificity: how well the parents identify causal influence
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import itertools

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])")
FINAL_ANSWER_PATTERN = re.compile(
    r"(?:the\s+answer\s+is|final\s+answer\s*(?:is|=)|answer\s*(?:is|=|:))\s*"
    r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
QUESTION_MARKER_PATTERN = re.compile(r"(?:\[\s*question\s*\]|question\s*[:\]\-])", re.IGNORECASE)


@dataclass
class InterventionResult:
    """Record of a single intervention test."""
    test_id: str                               # unique identifier
    pair_id: str                               # which pair was tested
    value_node_id: str                        # which node was truncated at
    truncation_token_idx: int                 # where truncation occurred
    intervention_type: str                    # "positive_control" or "negative_control"
    corruption_method: str                    # "false_values", "masking", "counterfactual"
    corrupted_node_ids: List[str]            # which nodes were corrupted
    baseline_answer: Optional[float]          # answer before intervention
    intervened_answer: Optional[float]        # answer after intervention
    answer_changed: bool                      # whether final answer changed
    confidence: float = 0.0                   # model confidence estimate (if available)
    generation_length: int = 0                # tokens generated after truncation
    errors: List[str] = field(default_factory=list)  # any errors during intervention


@dataclass
class CausalMetrics:
    """Aggregate metrics for causal validation."""
    total_tests: int = 0
    positive_control_tests: int = 0
    negative_control_tests: int = 0
    positive_hit_rate: float = 0.0            # % of positive controls with answer change
    negative_false_alarm_rate: float = 0.0    # % of negative controls with answer change
    specificity_score: float = 0.0            # (1 - false_alarm_rate)
    sensitivity_score: float = 0.0            # positive_hit_rate
    
    tests_by_method: Dict[str, int] = field(default_factory=dict)
    hit_rate_by_method: Dict[str, float] = field(default_factory=dict)


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
    return scratch_root / "traces" / model_name / experiment / "causal_validation_results.json"


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
    """Compute relative error between two values."""
    denominator = max(abs(expected), 1e-12)
    return abs(observed - expected) / denominator


def extract_numeric_spans(text: Optional[str]) -> List[Tuple[int, int, str]]:
    """Extract numeric spans: (start_char, end_char, value_text)."""
    if not text:
        return []
    spans = []
    for match in NUMBER_PATTERN.finditer(text):
        spans.append((match.start(), match.end(), match.group(0)))
    return spans


def truncate_after_first_question(text: str, truncation_idx: int) -> str:
    """Truncate text at given position, keeping only up to first question repeat."""
    truncated = text[:truncation_idx]
    # Check if second question appears after truncation point
    matches = list(QUESTION_MARKER_PATTERN.finditer(text))
    if len(matches) > 1 and matches[1].start() >= truncation_idx:
        return truncated
    return truncated


def find_value_token_position_in_trace(
    value_text: str,
    tokenizer: Any,
    trace_text: str,
) -> Optional[int]:
    """Find approximate token position of value_text in trace.
    
    Returns token index where value_text appears, or None if not found.
    """
    if value_text not in trace_text:
        return None
    
    # Find character position
    char_pos = trace_text.find(value_text)
    if char_pos == -1:
        return None
    
    # Tokenize and find which token contains this position
    try:
        encoded = tokenizer(trace_text, return_offsets_mapping=True, add_special_tokens=False)
        token_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        
        for tok_idx, (start, end) in enumerate(offsets):
            if start <= char_pos < end:
                return tok_idx
        return None
    except Exception:
        return None


def corrupt_node_in_trace(
    trace_text: str,
    node_id: str,
    value_text: str,
    value_token_range: Tuple[int, int],  # (start_char, end_char)
    corruption_method: str,
    replacement_value: Optional[str] = None,
) -> str:
    """Corrupt a single node's value occurrence in the trace."""
    start_char, end_char = value_token_range
    
    if corruption_method == "false_values":
        # Replace with a plausible but incorrect number
        if replacement_value is None:
            # Try to generate a false value based on the original
            try:
                orig_val = float(normalize_number_string(value_text))
                # Multiply by 2 or add noise
                false_val = orig_val * 2.0 + 0.5
                replacement_value = str(false_val)
            except:
                replacement_value = "0.0"
        return trace_text[:start_char] + replacement_value + trace_text[end_char:]
    
    elif corruption_method == "masking":
        # Replace with placeholder
        return trace_text[:start_char] + "[REDACTED]" + trace_text[end_char:]
    
    elif corruption_method == "counterfactual":
        # Replace with placeholder indicating uncertainty
        return trace_text[:start_char] + "$VALUE" + trace_text[end_char:]
    
    else:
        raise ValueError(f"Unknown corruption method: {corruption_method}")


def load_graph_json(graph_path: Path) -> Optional[Dict]:
    """Load graph.json from a patch run."""
    if not graph_path.exists():
        return None
    try:
        with open(graph_path) as f:
            return json.load(f)
    except Exception:
        return None


def load_aligned_pairs_dict(pairs_path: Path) -> Dict[str, Dict]:
    """Load aligned_pairs.json, indexed by pair_id."""
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
    """Extract base trace text from a pair record."""
    if not isinstance(pair, dict):
        return None
    
    nested_pair = pair.get("pair", {}) if isinstance(pair.get("pair"), dict) else {}
    nested_source = nested_pair.get("source", {}) if isinstance(nested_source.get("source"), dict) else {}
    
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
    """Extract expected answer from pair metadata."""
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


def generate_from_truncation(
    model: Any,
    tokenizer: Any,
    truncated_text: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
) -> Tuple[str, int]:
    """Generate continuation from truncated text.
    
    Returns (full_text, new_tokens_generated).
    """
    try:
        # Tokenize truncated text
        encoded = tokenizer(truncated_text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        
        # Generate
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
    except Exception as e:
        return truncated_text, 0


def generate_batch(
    model: Any,
    tokenizer: Any,
    truncated_texts: List[str],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
) -> List[Tuple[str, int]]:
    """Generate continuations for a batch of truncated texts in parallel.
    
    Returns list of (full_text, new_tokens_generated) tuples.
    """
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
    except Exception as e:
        # Fallback to sequential generation
        return [generate_from_truncation(model, tokenizer, text, max_new_tokens, temperature) for text in truncated_texts]


def run_intervention_test(
    model: Any,
    tokenizer: Any,
    pair_id: str,
    pair: Dict,
    graph: Dict,
    value_node_id: str,
    node_stats: Dict,
    corrupted_node_ids: List[str],
    control_type: str,  # "positive" or "negative"
    corruption_method: str,
    max_new_tokens: int = 50,
) -> InterventionResult:
    """Run a single intervention test.
    
    Args:
        model: HuggingFace language model
        tokenizer: HuggingFace tokenizer
        pair_id: identifier of the pair being tested
        pair: pair record from aligned_pairs.json
        graph: graph.json structure
        value_node_id: node ID where we're truncating (e.g., "v0")
        node_stats: stats for all nodes
        corrupted_node_ids: list of node IDs to corrupt ([] for baseline)
        control_type: "positive" (corrupt parents) or "negative" (corrupt non-parents)
        corruption_method: "false_values", "masking", or "counterfactual"
    """
    
    test_id = f"{pair_id}_{value_node_id}_{control_type}_{corruption_method}_{len(corrupted_node_ids)}"
    result = InterventionResult(
        test_id=test_id,
        pair_id=pair_id,
        value_node_id=value_node_id,
        truncation_token_idx=node_stats.get(value_node_id, {}).get("truncation_token_index", -1),
        intervention_type=f"{control_type}_control",
        corruption_method=corruption_method,
        corrupted_node_ids=corrupted_node_ids,
        baseline_answer=None,
        intervened_answer=None,
        answer_changed=False,
    )
    
    try:
        # Get baseline trace and answer
        trace_text = get_pair_trace_text(pair)
        expected_answer = get_pair_expected_answer(pair)
        
        if not trace_text:
            result.errors.append("No trace text found in pair")
            return result
        if expected_answer is None:
            result.errors.append("No expected answer in pair metadata")
            return result
        
        # Truncate at value_node_id
        trunc_idx = node_stats.get(value_node_id, {}).get("truncation_token_index", -1)
        if trunc_idx < 0:
            result.errors.append(f"Invalid truncation index for {value_node_id}")
            return result
        
        truncated_text = truncate_after_first_question(trace_text, trunc_idx)
        
        # Get baseline answer
        baseline_full, _ = generate_from_truncation(model, tokenizer, truncated_text, max_new_tokens)
        baseline_answer = extract_final_answer_value(baseline_full)
        result.baseline_answer = baseline_answer
        
        # Now corrupt nodes and re-generate
        corrupted_text = truncated_text
        for corrupt_node_id in corrupted_node_ids:
            # Find this node's value in the trace
            node_value_texts = [n.get("value_texts", []) for n in graph["nodes"] if n.get("id") == corrupt_node_id]
            if node_value_texts and node_value_texts[0]:
                value_text = node_value_texts[0][0]
                # Find this value in corrupted_text
                char_start = corrupted_text.find(value_text)
                if char_start >= 0:
                    char_end = char_start + len(value_text)
                    corrupted_text = corrupt_node_in_trace(
                        corrupted_text,
                        corrupt_node_id,
                        value_text,
                        (char_start, char_end),
                        corruption_method,
                    )
        
        # Generate from corrupted text
        intervened_full, gen_len = generate_from_truncation(model, tokenizer, corrupted_text, max_new_tokens)
        intervened_answer = extract_final_answer_value(intervened_full)
        result.intervened_answer = intervened_answer
        result.generation_length = gen_len
        
        # Check if answer changed
        if baseline_answer is not None and intervened_answer is not None:
            error = compute_relative_error(intervened_answer, baseline_answer)
            result.answer_changed = error > 0.05  # Small threshold for numerical variation
        
    except Exception as e:
        result.errors.append(str(e))
    
    return result


def run_validation_on_pair(
    model: Any,
    tokenizer: Any,
    pair_id: str,
    pair_path: Path,
    aligned_pairs_dict: Dict[str, Dict],
    max_tests_per_pair: int = 10,
    batch_size: int = 8,
) -> List[InterventionResult]:
    """Run validation tests on a single pair with batched generation."""
    
    results = []
    trace_text = None
    expected_answer = None
    
    # Load graph.json alongside pair
    graph_path = pair_path.parent / "graph.json"
    graph = load_graph_json(graph_path)
    if not graph:
        return results
    
    pair = aligned_pairs_dict.get(str(pair_id))
    if not pair:
        return results
    
    trace_text = get_pair_trace_text(pair)
    expected_answer = get_pair_expected_answer(pair)
    if not trace_text or expected_answer is None:
        return results
    
    node_stats = graph.get("node_stats", {})
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    
    # Build parent/child relationships
    edges = graph.get("edges", [])
    parents_of = defaultdict(set)  # child -> set of parents
    children_of = defaultdict(set)  # parent -> set of children
    
    for edge in edges:
        src, dst = edge["source"], edge["target"]
        parents_of[dst].add(src)
        children_of[src].add(dst)
    
    # Test a subset of nodes as truncation points
    test_nodes = [n for n in nodes.keys() if n != "prompt"]  # exclude prompt-value-only nodes
    if len(test_nodes) > 5:
        test_nodes = test_nodes[:5]  # limit to first 5 if many
    
    # Batch up all tests to run in parallel
    pending_tests: List[Tuple[InterventionResult, str, List[str], str]] = []  # (result_obj, truncated_text, corrupted_node_ids, control_type)
    
    for value_node_id in test_nodes:
        if len(results) >= max_tests_per_pair:
            break
            
        node_parents = list(parents_of.get(value_node_id, set()))
        all_other_nodes = [n for n in nodes.keys() if n != value_node_id and n not in node_parents]
        
        if not node_parents:
            continue  # Skip nodes with no parents
        
        # Positive control: corrupt parents
        for corruption_method in ["false_values", "masking"]:
            if len(results) + len(pending_tests) >= max_tests_per_pair:
                break
            
            corrupted_node_ids = node_parents[:min(2, len(node_parents))]
            trunc_idx = node_stats.get(value_node_id, {}).get("truncation_token_index", -1)
            if trunc_idx < 0:
                continue
            
            truncated_text = truncate_after_first_question(trace_text, trunc_idx)
            
            test_id = f"{pair_id}_{value_node_id}_positive_{corruption_method}_{len(corrupted_node_ids)}"
            result = InterventionResult(
                test_id=test_id,
                pair_id=pair_id,
                value_node_id=value_node_id,
                truncation_token_idx=trunc_idx,
                intervention_type="positive_control",
                corruption_method=corruption_method,
                corrupted_node_ids=corrupted_node_ids,
                baseline_answer=None,
                intervened_answer=None,
                answer_changed=False,
            )
            pending_tests.append((result, truncated_text, corrupted_node_ids, "positive"))
        
        # Negative control: corrupt non-parents
        if all_other_nodes:
            for corruption_method in ["false_values"]:
                if len(results) + len(pending_tests) >= max_tests_per_pair:
                    break
                
                non_parents = all_other_nodes[:min(2, len(all_other_nodes))]
                trunc_idx = node_stats.get(value_node_id, {}).get("truncation_token_index", -1)
                if trunc_idx < 0:
                    continue
                
                truncated_text = truncate_after_first_question(trace_text, trunc_idx)
                
                test_id = f"{pair_id}_{value_node_id}_negative_{corruption_method}_{len(non_parents)}"
                result = InterventionResult(
                    test_id=test_id,
                    pair_id=pair_id,
                    value_node_id=value_node_id,
                    truncation_token_idx=trunc_idx,
                    intervention_type="negative_control",
                    corruption_method=corruption_method,
                    corrupted_node_ids=non_parents,
                    baseline_answer=None,
                    intervened_answer=None,
                    answer_changed=False,
                )
                pending_tests.append((result, truncated_text, non_parents, "negative"))
    
    # Process pending tests in batches
    for batch_start in range(0, len(pending_tests), batch_size):
        batch_end = min(batch_start + batch_size, len(pending_tests))
        batch = pending_tests[batch_start:batch_end]
        
        # Generate baselines for this batch
        baseline_texts = [truncated for (_, truncated, _, _) in batch]
        baselines = generate_batch(model, tokenizer, baseline_texts, max_new_tokens=50)
        
        # Generate intervened versions
        intervened_texts = []
        for (_, truncated, corrupted_node_ids, _) in batch:
            corrupted_text = truncated
            for corrupt_node_id in corrupted_node_ids:
                node_value_texts = [n.get("value_texts", []) for n in graph["nodes"] if n.get("id") == corrupt_node_id]
                if node_value_texts and node_value_texts[0]:
                    value_text = node_value_texts[0][0]
                    char_start = corrupted_text.find(value_text)
                    if char_start >= 0:
                        char_end = char_start + len(value_text)
                        corrupted_text = corrupt_node_in_trace(
                            corrupted_text, corrupt_node_id, value_text,
                            (char_start, char_end), 
                            batch[batch_start].corruption_method,
                        )
            intervened_texts.append(corrupted_text)
        
        intervened = generate_batch(model, tokenizer, intervened_texts, max_new_tokens=50)
        
        # Update results
        for i, (result, truncated, corrupted_node_ids, control_type) in enumerate(batch):
            baseline_text, baseline_len = baselines[i]
            intervened_text, intervened_len = intervened[i]
            
            baseline_answer = extract_final_answer_value(baseline_text)
            intervened_answer = extract_final_answer_value(intervened_text)
            
            result.baseline_answer = baseline_answer
            result.intervened_answer = intervened_answer
            result.generation_length = intervened_len
            
            if baseline_answer is not None and intervened_answer is not None:
                error = compute_relative_error(intervened_answer, baseline_answer)
                result.answer_changed = error > 0.05
            
            results.append(result)
    
    return results


def aggregate_metrics(results: List[InterventionResult]) -> CausalMetrics:
    """Compute aggregate metrics from intervention results."""
    metrics = CausalMetrics(total_tests=len(results))
    
    if not results:
        return metrics
    
    positive_controls = [r for r in results if r.intervention_type == "positive_control"]
    negative_controls = [r for r in results if r.intervention_type == "negative_control"]
    
    metrics.positive_control_tests = len(positive_controls)
    metrics.negative_control_tests = len(negative_controls)
    
    # Hit rate: % of positive controls where answer changed
    if positive_controls:
        changed = sum(1 for r in positive_controls if r.answer_changed)
        metrics.positive_hit_rate = changed / len(positive_controls)
    
    # False alarm rate: % of negative controls where answer changed
    if negative_controls:
        changed = sum(1 for r in negative_controls if r.answer_changed)
        metrics.negative_false_alarm_rate = changed / len(negative_controls)
    
    metrics.sensitivity_score = metrics.positive_hit_rate
    metrics.specificity_score = 1.0 - metrics.negative_false_alarm_rate
    
    # Metrics by corruption method
    for method in ["false_values", "masking", "counterfactual"]:
        tests_with_method = [r for r in positive_controls if r.corruption_method == method]
        if tests_with_method:
            metrics.tests_by_method[method] = len(tests_with_method)
            hit_rate = sum(1 for r in tests_with_method if r.answer_changed) / len(tests_with_method)
            metrics.hit_rate_by_method[method] = hit_rate
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Validate causal structure by token-level intervention."
    )
    parser.add_argument("--model-name", type=str, default="Qwen2.5-72B", help="Model folder name.")
    parser.add_argument("--experiment", type=str, default="velocity", help="Experiment name.")
    parser.add_argument("--graph-dir", type=Path, default=None, help="Directory containing patch_runs with graphs.")
    parser.add_argument("--output-json", type=Path, default=None, help="Output results JSON.")
    parser.add_argument("--max-pairs", type=int, default=5, help="Max pairs to test.")
    parser.add_argument("--max-tests-per-pair", type=int, default=10, help="Max tests per pair.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generation.")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu).")
    parser.add_argument("--dtype", type=str, default="float16", help="Model dtype (float32 or float16).")
    
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
    
    # Find all patch run directories
    pair_dirs = sorted([d for d in graph_dir.iterdir() if d.is_dir()])[:args.max_pairs]
    
    # Also load aligned_pairs.json if available (for pair metadata)
    pairs_json = graph_dir.parent / "aligned_pairs.json"
    aligned_pairs_dict = load_aligned_pairs_dict(pairs_json)
    
    all_results: List[InterventionResult] = []
    
    for pair_dir in pair_dirs:
        pair_name = pair_dir.name
        graph_path = pair_dir / "graph.json"
        
        if not graph_path.exists():
            print(f"Skipping {pair_name}: no graph.json")
            continue
        
        print(f"Testing {pair_name}...")
        
        results = run_validation_on_pair(
            model, tokenizer,
            pair_name, pair_dir,
            aligned_pairs_dict,
            max_tests_per_pair=args.max_tests_per_pair,
            batch_size=args.batch_size,
        )
        all_results.extend(results)
        print(f"  {len(results)} tests completed")
    
    # Aggregate metrics
    metrics = aggregate_metrics(all_results)
    
    # Save results
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
    print(f"Total tests: {metrics.total_tests}")
    print(f"Positive control hit rate: {metrics.positive_hit_rate:.2%}")
    print(f"Negative control false alarm rate: {metrics.negative_false_alarm_rate:.2%}")
    print(f"Sensitivity: {metrics.sensitivity_score:.2%}, Specificity: {metrics.specificity_score:.2%}")


if __name__ == "__main__":
    main()
