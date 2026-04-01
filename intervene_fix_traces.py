"""
Verify and fix reasoning traces via API.

Pipeline role:
1. Load traces.json produced by generate_traces.py (or other trace JSON)
2. Ask an API to verify whether each trace is mathematically consistent
3. If incorrect, request a corrected full trace with valid intermediate values and answer
4. Tokenize fixed traces and save to JSON

Environment variables (optional):
    OPENAI_ENDPOINT   Full chat completion endpoint URL
    OPENAI_API_KEY    Bearer token
    OPENAI_MODEL      Model name sent to the API

Examples:
    python intervene_fix_traces.py --experiment velocity
    python intervene_fix_traces.py --input-json /path/to/traces.json
    python intervene_fix_traces.py --skip-api
"""

import argparse
import json
import os
import re
import time
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

from dotenv import load_dotenv
from transformers import AutoTokenizer


load_dotenv()
REPO_ROOT = Path(__file__).resolve().parent.parent
NUMBER_PATTERN = re.compile(r"(?<![\\w.])[-+]?(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?(?:[eE][-+]?\\d+)?(?![\\w.])")
UNIT_PATTERN = re.compile(r"^\\s*([A-Za-z%\\/\\^\\*\\-]+(?:\\/[A-Za-z%\\/\\^\\*\\-]+)?)")
QUESTION_MARKER_PATTERN = re.compile(
    r"(?:\[\s*question\s*\]|question\s*[:\]\-)])",
    re.IGNORECASE,
)
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
    return scratch_root / "traces" / model_name / experiment / "fixed_traces.json"


def resolve_default_tokenizer_path(model_name: str) -> Path:
    return REPO_ROOT / "models" / model_name


def split_trace_fields(trace: Dict[str, Any]) -> Tuple[str, Optional[int], Dict[str, Any], Optional[int], Optional[str]]:
    """Extract prompt text/metadata and existing generated text from a trace entry."""
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
    existing_text = trace.get("generated_text")

    if isinstance(trace.get("prompt_metadata"), dict):
        prompt_metadata = trace["prompt_metadata"]
    else:
        prompt_metadata = {k: v for k, v in trace.items() if k not in core_keys}

    return prompt_text, format_id, prompt_metadata, trace_id, existing_text


def build_verify_fix_prompt(
    prompt_text: str,
    prompt_metadata: Dict[str, Any],
    existing_trace: Optional[str],
) -> str:
    existing_block = existing_trace if existing_trace else ""
    return (
        "You are a strict verifier for math/physics reasoning traces.\n"
        "Task:\n"
        "1) Check whether the EXISTING_TRACE is consistent with the prompt and metadata.\n"
        "2) If incorrect at any step, correct the trace by patching in the valid intermediate values and final answer. The corrected trace MUST BE IDENTICAL to the original trace EXCEPT for the corrected values.\n"
        "3) If correct (Be reasonable with precision -- values within 5 percent are acceptable), return the original trace unchanged.\n\n"
        "Output format requirements (strict):\n"
        "- Return ONLY a JSON object, no markdown.\n"
        "- JSON keys: is_correct (boolean), corrected_trace (string), notes (string), values (array).\n"
        "For values array, include one object per numeric value that appears in corrected_trace:\n"
        "- value_text (string): exact substring in corrected_trace\n"
        "- normalized_value (string): canonical numeric form (no commas)\n"
        "- unit (string|null): inferred unit if obvious\n"
        "- role (string): one of given|intermediate|final|unknown\n"
        "- variable_name (string|null): optional variable symbol/name\n"
        "- span_start (int): 0-based inclusive char offset in corrected_trace\n"
        "- span_end (int): 0-based exclusive char offset in corrected_trace\n"
        "For the values array, include all numeric values in the trace, even if they were not given in the prompt metadata, to help with downstream analysis.\n\n"
        f"PROMPT_TEXT:\n{prompt_text}\n\n"
        f"PROMPT_METADATA:\n{json.dumps(prompt_metadata, ensure_ascii=True, sort_keys=True)}\n\n"
        f"EXISTING_TRACE:\n{existing_block}\n"
    )


def normalize_number_string(raw: str) -> Optional[str]:
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
        normalized = normalize_number_string(value_text)
        token_start, token_end = char_span_to_token_span(offsets, span_start, span_end)

        values.append(
            {
                "value_text": value_text,
                "normalized_value": normalized,
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


def sanitize_model_values(raw_values: Any, trace_text: Optional[str], tokenizer: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_values, list) or not trace_text:
        return []

    offsets = try_get_token_offsets(tokenizer, trace_text)
    sanitized: List[Dict[str, Any]] = []

    for raw in raw_values:
        if not isinstance(raw, dict):
            continue

        span_start = raw.get("span_start")
        span_end = raw.get("span_end")
        if not isinstance(span_start, int) or not isinstance(span_end, int):
            continue
        if span_start < 0 or span_end <= span_start or span_end > len(trace_text):
            continue

        value_text = trace_text[span_start:span_end]
        declared_value_text = raw.get("value_text")
        if isinstance(declared_value_text, str) and declared_value_text != value_text:
            continue

        normalized = raw.get("normalized_value")
        if not isinstance(normalized, str) or normalize_number_string(normalized) is None:
            normalized = normalize_number_string(value_text)

        token_start, token_end = char_span_to_token_span(offsets, span_start, span_end)
        role = raw.get("role") if raw.get("role") in {"given", "intermediate", "final", "unknown"} else "unknown"
        variable_name = raw.get("variable_name") if isinstance(raw.get("variable_name"), str) else None
        unit = raw.get("unit") if isinstance(raw.get("unit"), str) else infer_unit_after_span(trace_text, span_end)

        sanitized.append(
            {
                "value_text": value_text,
                "normalized_value": normalized,
                "unit": unit,
                "role": role,
                "variable_name": variable_name,
                "span_start": span_start,
                "span_end": span_end,
                "token_start": token_start,
                "token_end": token_end,
                "source": "model",
            }
        )

    return sanitized


def extract_numeric_values_from_metadata(prompt_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    values: List[Dict[str, Any]] = []
    for key, value in prompt_metadata.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            normalized = normalize_number_string(str(value))
            values.append(
                {
                    "key": key,
                    "value": value,
                    "normalized_value": normalized,
                }
            )
    return values


def truncate_after_first_question_block(text: Optional[str]) -> Tuple[Optional[str], bool]:
    """Truncate at the second 'Question:' marker to remove spillover into a new problem."""
    if not text:
        return text, False

    matches = list(QUESTION_MARKER_PATTERN.finditer(text))
    if len(matches) < 2:
        return text, False

    second_start = matches[1].start()
    truncated = text[:second_start].rstrip()
    return truncated, True


def parse_float_like(value: Any) -> Optional[float]:
    if value is None:
        return None
    normalized = normalize_number_string(str(value))
    if normalized is None:
        return None
    try:
        return float(normalized)
    except Exception:
        return None


def extract_expected_answer_from_metadata(prompt_metadata: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    for key, value in prompt_metadata.items():
        if not key.startswith("expected_"):
            continue
        expected = parse_float_like(value)
        if expected is not None:
            return key, expected
    return None, None


def extract_final_answer_value(text: Optional[str]) -> Optional[float]:
    if not text:
        return None

    matches = list(FINAL_ANSWER_PATTERN.finditer(text))
    if not matches:
        return None

    # Use the last explicit final-answer mention to avoid earlier intermediate mentions.
    last = matches[-1].group(1)
    return parse_float_like(last)


def compute_relative_error(observed: float, expected: float) -> float:
    denominator = max(abs(expected), 1e-12)
    return abs(observed - expected) / denominator


def call_api(
    api_url: str,
    api_key: Optional[str],
    api_model: str,
    user_prompt: str,
    timeout_s: int,
    temperature: float,
    max_tokens: int,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    payload = {
        "model": api_model,
        "messages": [
            {
                "role": "system",
                "content": "You verify and correct mathematical reasoning traces with strict consistency.",
            },
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(api_url, data=body, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        return None, None, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return None, None, str(e)

    content = None
    try:
        content = parsed["choices"][0]["message"]["content"]
    except Exception:
        content = None

    if not content:
        return None, parsed, "API response missing choices[0].message.content"

    return content.strip(), parsed, None


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of the first JSON object from model output."""
    if not text:
        return None

    # Fast path: full response is valid JSON.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Fallback: extract first {...} block.
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None

    return None


def tokenize_text(tokenizer: Any, text: Optional[str]) -> Tuple[List[int], List[str]]:
    if not text:
        return [], []
    token_ids = tokenizer(text, return_tensors=None)["input_ids"]
    token_strings = [tokenizer.decode([tid]) for tid in token_ids]
    return token_ids, token_strings



def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and fix traces via API.")
    parser.add_argument("--model-name", type=str, default="Qwen2.5-72B", help="Model folder name used for tokenizer and path defaults.")
    parser.add_argument("--experiment", type=str, default="velocity", help="Experiment name used in default paths.")
    parser.add_argument("--input-json", type=Path, default=None, help="Input traces JSON (default: traces.json from generate_traces.py).")
    parser.add_argument("--max-traces", type=int, default=None, help="Optional cap on number of loaded traces to process.")
    parser.add_argument("--output-json", type=Path, default=None, help="Output path for fixed traces JSON.")
    parser.add_argument("--tokenizer-path", type=Path, default=None, help="Tokenizer path for tokenizing corrected traces.")

    parser.add_argument("--api-url", type=str, default=os.getenv("OPENAI_ENDPOINT"), help="Chat completion endpoint URL.")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"), help="Bearer API key.")
    parser.add_argument("--api-model", type=str, default=os.getenv("OPENAI_MODEL"), help="Model name for API call.")
    parser.add_argument("--api-timeout", type=int, default=120, help="API request timeout in seconds.")
    parser.add_argument("--api-temperature", type=float, default=0.0, help="Temperature for API generation.")
    parser.add_argument("--api-max-tokens", type=int, default=1600, help="Max output tokens for API generation.")
    parser.add_argument("--max-relative-error", type=float, default=0.05, help="Maximum allowed relative error before API correction is triggered.")
    parser.add_argument("--request-sleep-seconds", type=float, default=0.0, help="Sleep between API requests.")
    parser.add_argument("--skip-api", action="store_true", help="Skip API calls and pass-through existing traces.")

    args = parser.parse_args()

    input_json = args.input_json or resolve_default_input_json(args.model_name, args.experiment)
    if not input_json.exists():
        raise FileNotFoundError(f"Input traces file not found: {input_json}")

    with open(input_json, "r") as f:
        loaded_traces = json.load(f)
    if not isinstance(loaded_traces, list):
        raise ValueError("Expected input JSON to be a list of trace objects")

    if args.max_traces is not None:
        if args.max_traces <= 0:
            raise ValueError("--max-traces must be > 0")
        loaded_traces = loaded_traces[: args.max_traces]

    output_json = args.output_json or resolve_default_output_json(args.model_name, args.experiment)
    tokenizer_path = args.tokenizer_path or resolve_default_tokenizer_path(args.model_name)

    # exit if json already there
    if output_json.exists():
        print(f"Output JSON already exists at {output_json}, exiting to avoid overwrite.")
        return

    if not args.skip_api and not args.api_url:
        raise ValueError("API URL is required unless --skip-api is set. Use --api-url or OPENAI_ENDPOINT.")

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    print(f"Loaded traces: {len(loaded_traces)}")

    traces_out: List[Dict[str, Any]] = []

    for out_id, source_trace in enumerate(loaded_traces):
        prompt_text, _format_id, prompt_metadata, source_trace_id, existing_text = split_trace_fields(source_trace)

        status = "skipped"
        error_message = None
        corrected_text = existing_text
        trace_numeric_values: List[Dict[str, Any]] = []
        expected_answer_key = None
        expected_answer_value = None
        extracted_answer_before_api = None
        relative_error_before_api = None
        
        corrected_text, truncation_applied = truncate_after_first_question_block(corrected_text)
        expected_answer_key, expected_answer_value = extract_expected_answer_from_metadata(prompt_metadata)
        extracted_answer_before_api = extract_final_answer_value(corrected_text)
        if expected_answer_value is not None and extracted_answer_before_api is not None:
            relative_error_before_api = compute_relative_error(extracted_answer_before_api, expected_answer_value)

        should_call_api = False
        if not args.skip_api:
            if expected_answer_value is None:
                should_call_api = True
            elif extracted_answer_before_api is None:
                should_call_api = True
            elif relative_error_before_api is not None and relative_error_before_api > args.max_relative_error:
                should_call_api = True
            else:
                status = "within_threshold"

        if should_call_api:
            verify_prompt = build_verify_fix_prompt(
                prompt_text=prompt_text,
                prompt_metadata=prompt_metadata,
                existing_trace=corrected_text,
            )
            response_text, _raw, error_message = call_api(
                api_url=args.api_url,
                api_key=args.api_key,
                api_model=args.api_model,
                user_prompt=verify_prompt,
                timeout_s=args.api_timeout,
                temperature=args.api_temperature,
                max_tokens=args.api_max_tokens,
            )

            if error_message is None:
                parsed = extract_first_json_object(response_text or "")
                if parsed is None:
                    status = "error"
                    error_message = "Could not parse verifier JSON output"
                else:
                    corrected_text = parsed.get("corrected_trace") or existing_text
                    corrected_text, truncation_applied_after = truncate_after_first_question_block(corrected_text)
                    truncation_applied = truncation_applied or truncation_applied_after
                    trace_numeric_values = sanitize_model_values(parsed.get("values"), corrected_text, tokenizer)
                    status = "ok"
            else:
                status = "error"

            if args.request_sleep_seconds > 0:
                time.sleep(args.request_sleep_seconds)

        print(f"\nTrace ID {source_trace_id} verification status: {status}")
        print(f"  original trace : {existing_text if existing_text else 'None'}")
        print(f"  corrected trace: {corrected_text if corrected_text else 'None'}")
        print(f"  expected answer: {expected_answer_value} (key: {expected_answer_key})")
        print(f"  extracted answer before API: {extracted_answer_before_api} with relative error {relative_error_before_api}")
        if error_message:
            print(f"  error message: {error_message}")

        if not trace_numeric_values:
            trace_numeric_values = extract_numeric_spans_from_text(corrected_text, tokenizer)
        token_ids, _token_strings = tokenize_text(tokenizer, corrected_text)
        prompt_token_ids, _prompt_token_strings = tokenize_text(tokenizer, prompt_text)

        traces_out.append(
            {
                "id": out_id,
                "trace_id": source_trace_id,
                "prompt": prompt_text,
                "prompt_metadata": prompt_metadata,
                "prompt_tokens": prompt_token_ids,
                "generated_text": corrected_text,
                "tokens": token_ids,
                "values": trace_numeric_values,
            }
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "traces": traces_out,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("=" * 80)
    print("TRACE VERIFICATION/FIX COMPLETE")
    print("=" * 80)
    print(f"Experiment: {args.experiment}")
    print(f"Input JSON: {input_json}")
    print(f"Processed traces: {len(traces_out)}")
    print(f"API calls enabled: {not args.skip_api}")
    print(f"Output JSON: {output_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
