#!/usr/bin/env python3
"""Strictly validate traces and keep only the ones with a correct answer.

Pipeline:
1. Load traces.json produced by generate_traces.py.
2. Truncate runaway completions so we only inspect the first trace segment.
3. Extract the final answer locally from the truncated text.
4. Compare against the expected answer in prompt metadata.
5. Reject traces whose answer is missing or outside the tolerance.
6. Save accepted traces to reject_traces.json for downstream pairing.

No API repair is attempted anymore.
"""

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])")
UNIT_PATTERN = re.compile(r"^\s*([A-Za-z%\/\^\*\-]+(?:\/[A-Za-z%\/\^\*\-]+)?)")
QUESTION_MARKER_PATTERN = re.compile(r"(?:\[\s*question\s*\]|question\s*[:\]\-])", re.IGNORECASE)
FINAL_ANSWER_PATTERN = re.compile(
    r"(?:the\s+answer\s+is|final\s+answer\s*(?:is|=)|answer\s*(?:is|=|:))\s*"
    r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


def resolve_default_input_json(model_name: str, experiment: str) -> Path:
    scratch_root = Path.home() / "links" / "scratch"
    if not scratch_root.exists():
        scratch_root = Path.home() / "scratch"
    return scratch_root / "traces" / model_name / experiment / "traces.json"


def resolve_default_output_json(model_name: str, experiment: str) -> Path:
    scratch_root = Path.home() / "links" / "scratch"
    if not scratch_root.exists():
        scratch_root = Path.home() / "scratch"
    return scratch_root / "traces" / model_name / experiment / "reject_traces.json"


def resolve_default_tokenizer_path(model_name: str) -> Path:
    return REPO_ROOT / "models" / model_name


def load_trace_list_from_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("traces"), list):
        return payload["traces"]

    raise ValueError("Expected input JSON to be a trace list or a payload containing top-level 'traces'")


def split_trace_fields(trace: Dict[str, Any]) -> Tuple[str, Optional[int], Dict[str, Any], Optional[int], Optional[str], List[int], List[str], Optional[int]]:
    core_keys = {
        "id",
        "prompt",
        "format_id",
        "tokens",
        "token_strings",
        "prompt_length",
        "generated_text",
        "prompt_metadata",
    }
    prompt_text = trace.get("prompt", "")
    format_id = trace.get("format_id")
    trace_id = trace.get("id")
    generated_text = trace.get("generated_text")
    tokens = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
    token_strings = trace.get("token_strings") if isinstance(trace.get("token_strings"), list) else []
    prompt_length = trace.get("prompt_length")

    if isinstance(trace.get("prompt_metadata"), dict):
        prompt_metadata = trace["prompt_metadata"]
    else:
        prompt_metadata = {k: v for k, v in trace.items() if k not in core_keys}

    return prompt_text, format_id, prompt_metadata, trace_id, generated_text, tokens, token_strings, prompt_length


def normalize_number_string(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    cleaned = str(raw).strip().replace(",", "")
    if not cleaned:
        return None
    try:
        dec = Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None

    normalized = format(dec, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"", "-0"}:
        normalized = "0"
    return normalized


def infer_unit_after_span(text: str, span_end: int) -> Optional[str]:
    if span_end >= len(text):
        return None
    match = UNIT_PATTERN.match(text[span_end:])
    if not match:
        return None
    unit = match.group(1).strip()
    return unit or None


def try_get_token_offsets(tokenizer: Any, text: str) -> Optional[List[Tuple[int, int]]]:
    try:
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    except Exception:
        return None

    offsets = encoded.get("offset_mapping") if isinstance(encoded, dict) else None
    if not isinstance(offsets, list):
        return None
    return [(int(start), int(end)) for start, end in offsets]


def char_span_to_token_span(offsets: Optional[List[Tuple[int, int]]], span_start: int, span_end: int) -> Tuple[Optional[int], Optional[int]]:
    if offsets is None:
        return None, None

    token_start = None
    token_end = None
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= span_start:
            continue
        if tok_start >= span_end:
            break
        if token_start is None:
            token_start = idx
        token_end = idx + 1
    return token_start, token_end


def extract_numeric_spans_from_text(text: Optional[str], tokenizer: Any) -> List[Dict[str, Any]]:
    if not text:
        return []

    offsets = try_get_token_offsets(tokenizer, text)
    values: List[Dict[str, Any]] = []

    for match in NUMBER_PATTERN.finditer(text):
        value_text = match.group(0)
        span_start, span_end = match.span()
        token_start, token_end = char_span_to_token_span(offsets, span_start, span_end)
        values.append(
            {
                "value_text": value_text,
                "normalized_value": normalize_number_string(value_text),
                "unit": infer_unit_after_span(text, span_end),
                "role": "unknown",
                "variable_name": None,
                "span_start": span_start,
                "span_end": span_end,
                "token_start": token_start,
                "token_end": token_end,
                "source": "regex",
            }
        )
    return values


def extract_expected_answer_from_metadata(prompt_metadata: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    for key, value in prompt_metadata.items():
        if not key.startswith("expected_"):
            continue
        normalized = normalize_number_string(value)
        if normalized is None:
            continue
        try:
            return key, float(normalized)
        except Exception:
            continue
    return None, None


def truncate_after_first_question_block(text: Optional[str]) -> Tuple[Optional[str], bool]:
    """Remove spillover into a second question block if the model keeps going."""
    if not text:
        return text, False

    matches = list(QUESTION_MARKER_PATTERN.finditer(text))
    if len(matches) < 2:
        return text, False

    second_start = matches[1].start()
    return text[:second_start].rstrip(), True


def extract_final_answer_value(text: Optional[str]) -> Optional[float]:
    if not text:
        return None

    matches = list(FINAL_ANSWER_PATTERN.finditer(text))
    if matches:
        candidate = matches[-1].group(1)
        normalized = normalize_number_string(candidate)
        return float(normalized) if normalized is not None else None

    return None


def extract_last_number_value(text: Optional[str]) -> Optional[float]:
    """Fallback extractor that uses the last numeric mention in the text."""
    if not text:
        return None

    matches = list(NUMBER_PATTERN.finditer(text))
    if not matches:
        return None

    candidate = matches[-1].group(0)
    normalized = normalize_number_string(candidate)
    return float(normalized) if normalized is not None else None


def extract_answer_value(text: Optional[str], fallback_to_last_number: bool = True) -> Tuple[Optional[float], str]:
    """Extract the most likely final answer from a completion.

    Prefer explicit final-answer phrasing, but fall back to the last numeric
    mention when the model gives a correct answer without the expected cue.
    """
    extracted = extract_final_answer_value(text)
    if extracted is not None:
        return extracted, "final_answer_pattern"

    if fallback_to_last_number:
        extracted = extract_last_number_value(text)
        if extracted is not None:
            return extracted, "last_number_fallback"

    return None, "missing_final_answer"


def compute_relative_error(observed: float, expected: float) -> float:
    denominator = max(abs(expected), 1e-12)
    return abs(observed - expected) / denominator


def tokenize_text(tokenizer: Any, text: Optional[str]) -> Tuple[List[int], List[str]]:
    if not text:
        return [], []
    token_ids = tokenizer(text, return_tensors=None)["input_ids"]
    token_strings = [tokenizer.decode([tid]) for tid in token_ids]
    return token_ids, token_strings


def build_output_trace(
    out_id: int,
    source_trace_id: Optional[int],
    prompt_text: str,
    prompt_metadata: Dict[str, Any],
    corrected_text: str,
    tokenizer: Any,
    validation: Dict[str, Any],
) -> Dict[str, Any]:
    token_ids, token_strings = tokenize_text(tokenizer, corrected_text)
    prompt_token_ids, prompt_token_strings = tokenize_text(tokenizer, prompt_text)
    values = extract_numeric_spans_from_text(corrected_text, tokenizer)

    return {
        "id": out_id,
        "trace_id": source_trace_id,
        "prompt": prompt_text,
        "prompt_metadata": prompt_metadata,
        "prompt_tokens": prompt_token_ids,
        "prompt_token_strings": prompt_token_strings,
        "prompt_length": len(prompt_token_ids),
        "generated_text": corrected_text,
        "tokens": token_ids,
        "token_strings": token_strings,
        "values": values,
        "verification": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly validate traces and keep only those with correct answers.")
    parser.add_argument("--model-name", type=str, default="Qwen2.5-72B", help="Model folder name used for tokenizer and path defaults.")
    parser.add_argument("--experiment", type=str, default="velocity", help="Experiment name used in default paths.")
    parser.add_argument("--input-json", type=Path, default=None, help="Input traces JSON (default: traces.json from generate_traces.py).")
    parser.add_argument("--max-traces", type=int, default=None, help="Optional cap on number of loaded traces to process.")
    parser.add_argument("--output-json", type=Path, default=None, help="Output path for reject traces JSON.")
    parser.add_argument("--tokenizer-path", type=Path, default=None, help="Tokenizer path for tokenizing accepted traces.")
    parser.add_argument("--max-relative-error", type=float, default=0.05, help="Maximum allowed relative error for acceptance.")
    parser.set_defaults(fallback_to_last_number=True)
    parser.add_argument(
        "--fallback-to-last-number",
        dest="fallback_to_last_number",
        action="store_true",
        help="Fallback to using the last numeric mention when explicit final-answer phrasing is missing (default: on).",
    )
    parser.add_argument(
        "--no-fallback-to-last-number",
        dest="fallback_to_last_number",
        action="store_false",
        help="Require explicit final-answer phrasing and disable the numeric fallback.",
    )

    args = parser.parse_args()

    input_json = args.input_json or resolve_default_input_json(args.model_name, args.experiment)
    if not input_json.exists():
        raise FileNotFoundError(f"Input traces file not found: {input_json}")

    traces = load_trace_list_from_json(input_json)
    if args.max_traces is not None:
        if args.max_traces <= 0:
            raise ValueError("--max-traces must be > 0")
        traces = traces[: args.max_traces]

    output_json = args.output_json or resolve_default_output_json(args.model_name, args.experiment)
    tokenizer_path = args.tokenizer_path or resolve_default_tokenizer_path(args.model_name)

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    print(f"Loaded input traces: {len(traces)}")

    accepted_traces: List[Dict[str, Any]] = []
    rejected_counts = {
        "missing_expected_answer": 0,
        "missing_final_answer": 0,
        "outside_tolerance": 0,
        "empty_trace": 0,
    }

    for out_id, source_trace in enumerate(traces):
        prompt_text, _format_id, prompt_metadata, source_trace_id, existing_text, _tokens, _token_strings, _prompt_length = split_trace_fields(source_trace)

        corrected_text, truncation_applied = truncate_after_first_question_block(existing_text)
        corrected_text = corrected_text or ""

        expected_answer_key, expected_answer_value = extract_expected_answer_from_metadata(prompt_metadata)
        extracted_answer, extraction_method = extract_answer_value(
            corrected_text,
            fallback_to_last_number=args.fallback_to_last_number,
        )

        accepted = False
        relative_error = None
        reason = "unknown"

        if not corrected_text.strip():
            rejected_counts["empty_trace"] += 1
            reason = "empty_trace"
        elif expected_answer_value is None:
            rejected_counts["missing_expected_answer"] += 1
            reason = "missing_expected_answer"
        elif extracted_answer is None:
            rejected_counts["missing_final_answer"] += 1
            reason = "missing_final_answer"
        else:
            relative_error = compute_relative_error(extracted_answer, expected_answer_value)
            if relative_error <= args.max_relative_error:
                accepted = True
                reason = "accepted"
            else:
                rejected_counts["outside_tolerance"] += 1
                reason = "outside_tolerance"

        print(
            f"Trace ID {source_trace_id}: {reason} | expected={expected_answer_value} ({expected_answer_key}) | "
            f"extracted={extracted_answer} | rel_error={relative_error} | truncated={truncation_applied}"
        )

        if not accepted:
            continue

        validation = {
            "expected_answer_key": expected_answer_key,
            "expected_answer_value": expected_answer_value,
            "extracted_answer": extracted_answer,
            "extraction_method": extraction_method,
            "relative_error": relative_error,
            "max_relative_error": args.max_relative_error,
            "truncation_applied": truncation_applied,
            "accepted": True,
        }

        accepted_traces.append(
            build_output_trace(
                out_id=len(accepted_traces),
                source_trace_id=source_trace_id,
                prompt_text=prompt_text,
                prompt_metadata=prompt_metadata,
                corrected_text=corrected_text,
                tokenizer=tokenizer,
                validation=validation,
            )
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "traces": accepted_traces,
        "summary": {
            "input_traces": len(traces),
            "output_traces": len(accepted_traces),
            "success_rate": (len(accepted_traces) / len(traces)) if traces else 0.0,
            "max_relative_error": args.max_relative_error,
            "rejected_counts": rejected_counts,
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    success_rate = (len(accepted_traces) / len(traces)) if traces else 0.0
    print("=" * 80)
    print("TRACE VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Input traces : {len(traces)}")
    print(f"Output traces: {len(accepted_traces)}")
    print(f"Success rate : {success_rate:.4f}")
    print(f"Output JSON  : {output_json}")
    print("Rejected counts:")
    for key, value in rejected_counts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()