"""
Generate and persist prompt pairs for causal patching experiments.

This script creates a stable list of source/base trace pairs and writes them
to JSON so downstream scripts can reuse exactly the same pairs across runs.

Examples:
  python intervene_generate_pairs.py
  python intervene_generate_pairs.py --trace-indices 0 1 2 3 10 11 12 13
  python intervene_generate_pairs.py --indices-json /path/to/indices.json
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


TRACES_METADATA_FILE = Path(
    "/home/wuroderi/links/scratch/reasoning_traces/Qwen2.5-32B/velocity/traces_metadata.json"
)
DEFAULT_OUTPUT_JSON = Path(
    "/home/wuroderi/links/scratch/reasoning_traces/Qwen2.5-32B/velocity/prompt_pairs.json"
)


def build_pairs_from_traces(traces: List[Dict], n_prompt_pairs: int = None) -> List[Dict]:
    traces_by_format = defaultdict(list)
    for trace in traces:
        traces_by_format[trace["format_id"]].append(trace)

    if not traces_by_format:
        return []

    min_traces_per_format = min(len(v) for v in traces_by_format.values())
    n_pairs_per_format = min_traces_per_format // 2

    pairs = []
    for format_id in sorted(traces_by_format.keys()):
        format_traces = traces_by_format[format_id]

        if n_prompt_pairs is None:
            pairs_to_create = n_pairs_per_format
        else:
            per_format = n_prompt_pairs // len(traces_by_format) + 1
            pairs_to_create = min(n_pairs_per_format, per_format)

        for i in range(pairs_to_create):
            if n_prompt_pairs is not None and len(pairs) >= n_prompt_pairs:
                break

            source_trace = format_traces[i]
            base_trace = format_traces[i + n_pairs_per_format]

            pairs.append((format_id, source_trace, base_trace))

        if n_prompt_pairs is not None and len(pairs) >= n_prompt_pairs:
            break

    return pairs


def truncate_at_second_question(text: str) -> str:
    """Keep content only up to the second occurrence of 'Question'."""
    matches = list(re.finditer(r"Question", text))
    if len(matches) < 2:
        return text
    return text[: matches[1].start()].rstrip()


def format_value_like_reference(reference: str, value: float) -> str:
    """
    Format a numeric value using the same textual style as a reference string.
    IMPORTANT: Returns a string with the SAME CHARACTER LENGTH as reference
    by padding with zeros, adjusting decimal places, or falling back to scientific notation.
    """
    reference = str(reference).strip()
    reference_len = len(reference)
    
    if "e" in reference.lower():
        # Scientific notation: keep the same format
        mantissa, exponent = re.split(r"[eE]", reference)
        if "." in mantissa:
            decimal_places = len(mantissa.split(".")[1])
        else:
            decimal_places = 0
        formatted = f"{value:.{decimal_places}e}"
        # Preserve case (E vs e)
        if "E" in reference:
            formatted = formatted.upper()
        # Pad/truncate to match reference length
        if len(formatted) < reference_len:
            # Add zeros to mantissa
            formatted = formatted.replace("e", "0" * (reference_len - len(formatted)) + "e")
        return formatted[:reference_len]  # Ensure exact length
    
    if "." in reference:
        # Decimal format: try to match decimal places and total length
        reference_parts = reference.split(".")
        reference_decimals = len(reference_parts[1])
        
        # Format with same decimal places
        formatted = f"{value:.{reference_decimals}f}"
        
        # If fits, pad with leading zeros if needed
        if len(formatted) <= reference_len:
            if len(formatted) < reference_len:
                formatted = formatted.zfill(reference_len)
            return formatted[:reference_len]
        
        # If too long, try reducing decimal places
        for decimals in range(reference_decimals - 1, -1, -1):
            formatted = f"{value:.{decimals}f}"
            if len(formatted) <= reference_len:
                if len(formatted) < reference_len:
                    formatted = formatted.zfill(reference_len)
                return formatted[:reference_len]
        
        # Last resort: use scientific notation with same length
        for decimals in range(5, -1, -1):
            formatted = f"{value:.{decimals}e}".upper() if "E" in reference or "e" in reference else f"{value:.{decimals}e}"
            if len(formatted) <= reference_len:
                if len(formatted) < reference_len:
                    # Pad mantissa part
                    parts = formatted.split("e" if "e" in formatted else "E")
                    mantissa, exp = parts[0], parts[1]
                    padding_needed = reference_len - len(formatted)
                    mantissa = mantissa.ljust(len(mantissa) + padding_needed, "0")
                    formatted = f"{mantissa}e{exp}" if "e" in formatted else f"{mantissa}E{exp}"
                return formatted[:reference_len]
        
        # If still too long, truncate (shouldn't happen)
        return formatted[:reference_len]
    
    # Integer format: match length by padding with leading zeros
    formatted = f"{value:.0f}"
    if len(formatted) < reference_len:
        formatted = formatted.zfill(reference_len)
    return formatted[:reference_len]


def replace_formatted_value_occurrences(
    text: str,
    old_value: float,
    new_value: float,
    offset: int = 0,
    allow_short_int: bool = False,
) -> Tuple[str, List[Tuple[int, int, str, str]]]:
    # Match numeric values, but be more permissive with trailing punctuation
    # Allows periods, commas, etc. after numbers (sentence endings)
    numeric_pattern = re.compile(r"(?<![\w])[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][+-]?\d+)?(?![\w])")

    # Skip replacement if old_value would match too many generic short numbers
    # (e.g., don't try to replace "17" or smaller integers because they're too generic)
    old_str = str(old_value).strip()
    if not allow_short_int and len(old_str) <= 2 and old_str.isdigit():
        return text, []  # Skip short integer matches to avoid false positives

    replacements = []
    result_parts = []
    last_end = 0

    for match in numeric_pattern.finditer(text):
        token = match.group()
        
        # Skip single-digit tokens - they're too likely to be false matches
        # (e.g., "2" in "v^2", "1" in "1/2mv^2", etc.)
        if not allow_short_int and len(token) == 1 and token.isdigit():
            continue
        
        if format_value_like_reference(token, old_value) != token:
            continue

        replacement = format_value_like_reference(token, new_value)
        result_parts.append(text[last_end:match.start()])
        result_parts.append(replacement)
        replacements.append((offset + match.start(), offset + match.end(), token, replacement))
        last_end = match.end()

    if not replacements:
        return text, []

    result_parts.append(text[last_end:])
    return "".join(result_parts), replacements


def extract_calculated_values_from_cot(text: str) -> Dict[str, float]:
    """
    Extract calculated v, t, v^2 values from model's CoT output.
    For t specifically, handles complex formulas like "t = d/v = 30/52.8 = 0.57 s"
    by finding the last numeric value before the unit 's'.
    
    Returns dict with keys: 'v', 't', 'v_squared' if found, else empty dict.
    """
    values = {}
    
    # v^2 extraction: collect all candidates and take the last one in the CoT
    # so we prefer later "computed" assignments over early setup equations.
    v2_candidates = []
    for m in re.finditer(r'([-+]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*=\s*[vV]\s*\^\s*2', text):
        v2_candidates.append((m.start(), m.group(1)))
    for m in re.finditer(r'[vV]\s*\^\s*2\s*=\s*([-+]?\d+\.?\d*(?:[eE][+-]?\d+)?)', text):
        start = m.start()
        # Ignore setup forms like "... kg)v^2 = ..." or "* v^2 = ...".
        prev_non_space = None
        for i in range(start - 1, -1, -1):
            if not text[i].isspace():
                prev_non_space = text[i]
                break
        if prev_non_space is not None and (prev_non_space.isalnum() or prev_non_space in {')', '*', '/'}):
            continue
        v2_candidates.append((start, m.group(1)))
    if v2_candidates:
        _, v2_token = max(v2_candidates, key=lambda x: x[0])
        try:
            values['v_squared'] = float(v2_token)
        except ValueError:
            pass
    
    # Pattern 1 for v: "number = v" but NOT followed by "^"
    v_match = re.search(r'([-+]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*=\s*[vV](?!\s*\^)(?:\s|\.)', text)
    if v_match:
        try:
            values['v'] = float(v_match.group(1))
        except ValueError:
            pass
    
    # Pattern 2 for v: "v = number" (for source format like "v = 52.8")
    if 'v' not in values:
        v_match = re.search(r'[vV]\s*=\s*([-+]?\d+\.?\d*(?:[eE][+-]?\d+)?)(?:\s|[a-zA-Z])', text)
        if v_match:
            try:
                values['v'] = float(v_match.group(1))
            except ValueError:
                pass
    
    # Pattern 1 for t: "number = t" or "number = t."
    t_match = re.search(r'([-+]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*=\s*[tT](?:\s|\.)', text)
    if t_match:
        try:
            values['t'] = float(t_match.group(1))
        except ValueError:
            pass
    
    # Pattern 2 for t: "t = <number> s" (with explicit 's' unit) - this handles formulas
    # by looking for the number immediately before/around the 's' unit
    if 't' not in values:
        # Match "= <number> s" without requiring "t =" (so it finds the final result)
        t_match = re.search(r'=\s*([-+]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*[sS](?:\s|\.)', text)
        if t_match:
            try:
                values['t'] = float(t_match.group(1))
            except ValueError:
                pass
    
    # Pattern 3 for t: Look for "t = number but no formula operators after it"
    if 't' not in values:
        # Find "t = " followed by a simple number (not complex expression)
        t_matches = list(re.finditer(r'[tT]\s*=\s*([-+]?\d+\.?\d*(?:[eE][+-]?\d+)?)(?=[\s;.]|$)', text))
        if t_matches:
            # Take the last one (most likely the final answer)
            try:
                values['t'] = float(t_matches[-1].group(1))
            except ValueError:
                pass
    
    # Pattern for final answer: "The answer is <number>"
    answer_match = re.search(r'[Tt]he\s+answer\s+is\s+([-+]?\d+\.?\d*(?:[eE][+-]?\d+)?)', text)
    if answer_match:
        try:
            values['answer'] = float(answer_match.group(1))
        except ValueError:
            pass
    
    return values


def merge_counterfactual_cot(
    base_text: str,
    source_text: str,
    source_ke: float,
    base_ke: float,
    source_mass: float,
    base_mass: float,
    source_d: float = None,
    base_d: float = None,
) -> Tuple[str, List[Tuple[int, int, str, str]]]:
    """
    Merge base text with source values by intelligently extracting and swapping
    what the model actually calculated in each CoT.
    
    Args:
        base_text: Full base prompt + CoT
        source_text: Full source prompt + CoT (for extracting source's calculated v, t, v^2)
        source_ke, base_ke: Kinetic energy values
        source_mass, base_mass: Mass values
        source_d, base_d: Distance values
    """
    # Extract what each model actually calculated
    base_cot = base_text.split("Answer (step-by-step): ")[-1] if "Answer (step-by-step): " in base_text else base_text
    source_cot = source_text.split("Answer (step-by-step): ")[-1] if "Answer (step-by-step): " in source_text else source_text
    
    base_values = extract_calculated_values_from_cot(base_cot)
    source_values = extract_calculated_values_from_cot(source_cot)

    replacements = []
    value_swaps = []

    # v^2: swap base's calculated v^2 with source's calculated v^2
    if 'v_squared' in base_values and 'v_squared' in source_values:
        value_swaps.append((base_values['v_squared'], source_values['v_squared'], "v^2"))
    elif 'v_squared' in base_values:
        # Compute source's v^2
        source_v2 = (2 * source_ke) / source_mass
        value_swaps.append((base_values['v_squared'], source_v2, "v^2"))
    
    # v: swap base's calculated v with source's calculated v
    if 'v' in base_values and 'v' in source_values:
        value_swaps.append((base_values['v'], source_values['v'], "v"))
    elif 'v' in base_values:
        # Compute source's v
        source_v = ((2 * source_ke) / source_mass) ** 0.5
        value_swaps.append((base_values['v'], source_v, "v"))

    # t: swap base's calculated t with source's calculated t
    if 't' in base_values and 't' in source_values:
        value_swaps.append((base_values['t'], source_values['t'], "t"))
    
    # 2*t: if we have t values
    if 't' in base_values and 't' in source_values:
        value_swaps.append((2 * base_values['t'], 2 * source_values['t'], "2*t"))

    # answer: swap base's final answer with source's final answer
    if 'answer' in base_values and 'answer' in source_values:
        value_swaps.append((base_values['answer'], source_values['answer'], "answer"))

    # d (distance) from trace values
    if source_d is not None and base_d is not None:
        value_swaps.append((base_d, source_d, "d"))

    # KE and mass (do last to avoid partial match issues)
    value_swaps.append((base_ke, source_ke, "KE"))
    value_swaps.append((base_mass, source_mass, "mass"))

    def swap_in_text(text: str, offset: int = 0) -> Tuple[str, List[Tuple[int, int, str, str]]]:
        """Replace all flagged values in text with formatting preservation."""
        local_replacements = []
        current_text = text
        
        # Sort value_swaps by length (longest first) to avoid partial matches
        sorted_swaps = sorted(value_swaps, key=lambda x: -len(str(x[0])))

        for old_val, new_val, label in sorted_swaps:
            current_text, val_replacements = replace_formatted_value_occurrences(
                current_text,
                old_value=old_val,
                new_value=new_val,
                offset=offset,
                allow_short_int=(label in {"d", "mass"}),
            )
            local_replacements.extend(val_replacements)

        return current_text, local_replacements

    # Swap in base text (not source)
    result_text = base_text
    result_text, all_replacements = swap_in_text(result_text, offset=0)
    replacements.extend(all_replacements)

    return result_text, replacements


def compact_trace(trace: Dict, trace_index: int, generated_text: str) -> Dict:
    return {
        "trace_index": trace_index,
        "trace_id": trace["id"],
        "format_id": trace["format_id"],
        "generated_text": generated_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stable prompt pairs JSON.")
    parser.add_argument(
        "--traces-metadata",
        type=Path,
        default=TRACES_METADATA_FILE,
        help="Path to traces_metadata.json.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output path for prompt pairs JSON.",
    )
    parser.add_argument(
        "--n-prompt-pairs",
        type=int,
        default=None,
        help="Optional cap on total number of pairs.",
    )
    parser.add_argument(
        "--trace-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional subset of trace list indices (0-based, from metadata order). "
            "Useful for selecting specific pre-generated trace_* entries."
        ),
    )
    parser.add_argument(
        "--indices-json",
        type=Path,
        default=None,
        help="Optional JSON file containing a list of trace indices to keep.",
    )
    args = parser.parse_args()

    with open(args.traces_metadata, "r") as f:
        all_traces = json.load(f)

    selected_indices = None
    if args.indices_json is not None:
        with open(args.indices_json, "r") as f:
            selected_indices = json.load(f)
    elif args.trace_indices is not None:
        selected_indices = args.trace_indices

    if selected_indices is not None:
        selected_set = {int(i) for i in selected_indices}
        filtered_traces = [tr for i, tr in enumerate(all_traces) if i in selected_set]
    else:
        filtered_traces = all_traces

    raw_pairs = build_pairs_from_traces(filtered_traces, n_prompt_pairs=args.n_prompt_pairs)

    index_map = {id(trace): i for i, trace in enumerate(all_traces)}
    pairs = []
    for pair_id, (format_id, source_trace, base_trace) in enumerate(raw_pairs):
        source_text = truncate_at_second_question(source_trace.get("generated_text", ""))
        base_text = truncate_at_second_question(base_trace.get("generated_text", ""))
        source_cot_only = source_text.split("Answer (step-by-step): ")[-1]
        base_cot_only = base_text.split("Answer (step-by-step): ")[-1]
        source_calc = extract_calculated_values_from_cot(source_cot_only)
        base_calc = extract_calculated_values_from_cot(base_cot_only)

        # Create counterfactuals:
        # 1. source text with trace_1 values (swap in trace_1's calculated v, t, etc.)
        merged_0_with_1, _ = merge_counterfactual_cot(
            base_text=source_text,
            source_text=base_text,
            source_ke=base_trace["ke"],
            base_ke=source_trace["ke"],
            source_mass=base_trace["m"],
            base_mass=source_trace["m"],
            source_d=base_trace.get("d"),
            base_d=source_trace.get("d"),
        )
        
        # 2. base text with trace_0 values (swap in trace_0's calculated v, t, etc.)
        merged_1_with_0, _ = merge_counterfactual_cot(
            base_text=base_text,
            source_text=source_text,
            source_ke=source_trace["ke"],
            base_ke=base_trace["ke"],
            source_mass=source_trace["m"],
            base_mass=base_trace["m"],
            source_d=source_trace.get("d"),
            base_d=base_trace.get("d"),
        )

        pair = {
            "pair_id": pair_id,
            "values": {
                "trace_0": {
                    "m": source_trace["m"],
                    "ke": source_trace["ke"],
                    "v": source_trace.get("v"),
                    "d": source_trace.get("d"),
                    "t": source_calc.get("answer", source_calc.get("t")),
                },
                "trace_1": {
                    "m": base_trace["m"],
                    "ke": base_trace["ke"],
                    "v": base_trace.get("v"),
                    "d": base_trace.get("d"),
                    "t": base_calc.get("answer", base_calc.get("t")),
                },
            },
            "cot_0": {
                "original": source_text,
                "with_1_values": merged_0_with_1,
            },
            "cot_1": {
                "original": base_text,
                "with_0_values": merged_1_with_0,
            },
        }
        pairs.append(pair)

    payload = {
        "schema_version": 5,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "input": {
            "traces_metadata": str(args.traces_metadata),
            "n_total_traces": len(all_traces),
            "n_selected_traces": len(filtered_traces),
            "selected_trace_indices": selected_indices,
            "n_prompt_pairs_requested": args.n_prompt_pairs,
        },
        "n_pairs": len(pairs),
        "description": f"Each pair has trace_0 and trace_1 with original CoTs and merged variants. trace_0.with_1_values has trace_0's CoT with trace_1's calculated values. trace_1.with_0_values has trace_1's CoT with trace_0's calculated values. Total: {len(pairs)} pairs × 2 directions = {len(pairs) * 2} patching experiments.",
        "pairs": pairs,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("=" * 80)
    print("PROMPT PAIRS GENERATED")
    print("=" * 80)
    print(f"Traces metadata: {args.traces_metadata}")
    print(f"Total traces: {len(all_traces)}")
    print(f"Selected traces: {len(filtered_traces)}")
    print(f"Generated pairs: {len(pairs)}")
    print(f"Patching directions: {len(pairs)} pairs × 2 directions = {len(pairs) * 2} experiments")
    print(f"Schema: v5 (trace_0/trace_1 with cot_0/cot_1, each with original + merged variants)")
    print(f"Output JSON: {args.output_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()