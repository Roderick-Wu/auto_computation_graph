"""
Generate counterfactual paired traces from fixed traces JSON.

Pipeline role:
1. Load fixed_traces.json produced by intervene_fix_traces.py
2. Sample a counterfactual metadata tuple from the same prompt generator family
3. Ask an API to rewrite the base trace with counterfactual values only
4. Tokenize generated text and extract numeric value spans for downstream graphing
"""

import argparse
import json
import os
import random
import re
import time
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

from dotenv import load_dotenv
from transformers import AutoTokenizer

import prompts


load_dotenv()
REPO_ROOT = Path(__file__).resolve().parent.parent

NUMBER_PATTERN = re.compile(r"(?<![\\w.])[-+]?(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?(?:[eE][-+]?\\d+)?(?![\\w.])")
UNIT_PATTERN = re.compile(r"^\\s*([A-Za-z%\\/\\^\\*\\-]+(?:\\/[A-Za-z%\\/\\^\\*\\-]+)?)")
QUESTION_MARKER_PATTERN = re.compile(r"(?:\[\s*question\s*\]|question\s*[:\]\-)])", re.IGNORECASE)


def is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def resolve_default_traces_json(model_name: str, experiment: str) -> Path:
    scratch_root = Path.home() / "links" / "scratch"
    if not scratch_root.exists():
        scratch_root = Path.home() / "scratch"
    return scratch_root / "traces" / model_name / experiment / "fixed_traces.json"


def resolve_default_output_json(model_name: str, experiment: str) -> Path:
    scratch_root = Path.home() / "links" / "scratch"
    if not scratch_root.exists():
        scratch_root = Path.home() / "scratch"
    return scratch_root / "traces" / model_name / experiment / "paired_traces.json"


def resolve_default_tokenizer_path(model_name: str) -> Path:
    return REPO_ROOT / "models" / model_name


def load_trace_list_from_json(path: Path) -> List[Dict[str, Any]]:
    """Load trace list from fixed_traces payload (dict containing 'traces')."""
    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and isinstance(payload.get("traces"), list):
        return payload["traces"]

    raise ValueError("Expected fixed_traces JSON payload containing a top-level 'traces' list")


def get_prompt_experiment_name(experiment: str) -> str:
    experiment_mapping = {
        "velocity": "velocity_from_ke",
        "velocity_uniform_t": "velocity_from_ke_uniform_t",
        "current": "current_from_power",
        "radius": "radius_from_area",
        "side_length": "side_length_from_volume",
        "wavelength": "wavelength_from_speed",
        "cross_section": "cross_section_from_flow",
        "displacement": "displacement_from_spring",
        "market_cap": "market_cap_from_shares",
    }

    if experiment in experiment_mapping:
        return experiment_mapping[experiment]

    generators = prompts.get_all_generators()
    if experiment in generators:
        return experiment

    available = sorted(list(experiment_mapping.keys()) + list(generators.keys()))
    raise ValueError(f"Unknown experiment: {experiment}. Available: {available}")


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
        "verification",
        "numeric_values",
        "values",
        "original_generated_text",
        "prompt_tokens",
        "prompt_token_strings",
    }
    prompt_text = trace.get("prompt", "")
    format_id = trace.get("format_id")
    trace_id = trace.get("id")
    source_generated_text = trace.get("generated_text")
    source_tokens = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
    source_token_strings = trace.get("token_strings") if isinstance(trace.get("token_strings"), list) else []
    source_prompt_length = trace.get("prompt_length")

    if isinstance(trace.get("prompt_metadata"), dict):
        prompt_metadata = trace["prompt_metadata"]
    else:
        prompt_metadata = {k: v for k, v in trace.items() if k not in core_keys}

    return (
        prompt_text,
        format_id,
        prompt_metadata,
        trace_id,
        source_generated_text,
        source_tokens,
        source_token_strings,
        source_prompt_length,
    )


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
    return match.group(1).strip() or None


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


def truncate_after_first_question_block(text: Optional[str]) -> Tuple[Optional[str], bool]:
    if not text:
        return text, False

    matches = list(QUESTION_MARKER_PATTERN.finditer(text))
    if len(matches) < 2:
        return text, False

    second_start = matches[1].start()
    return text[:second_start].rstrip(), True


def build_counterfactual_metadata(
    prompt_experiment_name: str,
    format_id: Optional[int],
    original_prompt_metadata: Dict[str, Any],
    rng: random.Random,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Sample a counterfactual tuple by reusing prompts.py generators for the same experiment/format."""
    candidates = prompts.generate_prompts_for_experiment(prompt_experiment_name, samples_per_format=8)

    filtered = [c for c in candidates if format_id is None or c.get("format_id") == format_id]
    if not filtered:
        filtered = candidates

    chosen = rng.choice(filtered)

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
    sampled_metadata = {k: v for k, v in chosen.items() if k not in core_keys}

    # Preserve all non-numeric fields from original metadata (e.g., object nouns).
    # Only numeric fields are counterfactually resampled.
    cf_metadata: Dict[str, Any] = dict(original_prompt_metadata)
    for key, sampled_value in sampled_metadata.items():
        if key in original_prompt_metadata and is_numeric_scalar(sampled_value):
            cf_metadata[key] = sampled_value

    replacements: List[Dict[str, Any]] = []
    for key, old_value in original_prompt_metadata.items():
        if key not in cf_metadata:
            continue
        new_value = cf_metadata[key]
        if old_value == new_value:
            continue

        if is_numeric_scalar(old_value) and is_numeric_scalar(new_value):
            replacements.append(
                {
                    "key": key,
                    "old": old_value,
                    "new": new_value,
                    "old_normalized": normalize_number_string(old_value),
                    "new_normalized": normalize_number_string(new_value),
                }
            )

    return cf_metadata, replacements


def build_generation_prompt(
    prompt_text: str,
    base_trace: Optional[str],
    original_prompt_metadata: Dict[str, Any],
    counterfactual_prompt_metadata: Dict[str, Any],
    replacements: List[Dict[str, Any]],
) -> str:
    base_trace_block = base_trace or ""

    return (
        "You rewrite math/physics traces under counterfactual values.\n"
        "Task:\n"
        "1) Start from EXISTING_TRACE exactly.\n"
        "2) Modify ONLY numbers/related variable values needed to make the trace (question and response) consistent with COUNTERFACTUAL_PROMPT_METADATA. Don't forget to update the ENTIRE trace -- both the question and response.\n"
        "3) Do NOT change non-numeric words in the trace (objects/entities must remain identical).\n"
        "4) Keep style, wording, and step structure IDENTICAL to the EXISTING_TRACE.\n"
        "Output format requirements (strict):\n"
        "- Return ONLY a JSON object, no markdown.\n"
        "- JSON keys: generated_trace (string), notes (string).\n"
        f"PROMPT_TEXT:\n{prompt_text}\n\n"
        f"EXISTING_TRACE:\n{base_trace_block}\n\n"
        f"ORIGINAL_PROMPT_METADATA:\n{json.dumps(original_prompt_metadata, ensure_ascii=True, sort_keys=True)}\n\n"
        f"COUNTERFACTUAL_PROMPT_METADATA:\n{json.dumps(counterfactual_prompt_metadata, ensure_ascii=True, sort_keys=True)}\n\n"
        f"SUGGESTED_REPLACEMENTS:\n{json.dumps(replacements, ensure_ascii=True, sort_keys=True)}\n"
    )


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
            {"role": "system", "content": "You produce precise math and physics reasoning traces."},
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
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

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
    parser = argparse.ArgumentParser(description="Generate paired source/counterfactual traces from fixed_traces.json.")
    parser.add_argument("--model-name", type=str, default="Qwen2.5-72B", help="Model folder name used for tokenizer and output path.")
    parser.add_argument("--experiment", type=str, default="velocity", help="Experiment name used in default trace/output paths.")
    parser.add_argument("--traces-json", type=Path, default=None, help="Path to fixed_traces.json produced by intervene_fix_traces.py.")
    parser.add_argument("--max-traces", type=int, default=None, help="Optional cap on number of loaded traces to process.")
    parser.add_argument("--output-json", type=Path, default=None, help="Output path for paired traces JSON.")
    parser.add_argument("--tokenizer-path", type=Path, default=None, help="Tokenizer path for tokenizing generated traces.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for counterfactual sampling.")

    parser.add_argument("--api-url", type=str, default=os.getenv("OPENAI_ENDPOINT"), help="Chat completion endpoint URL.")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"), help="Bearer API key.")
    parser.add_argument("--api-model", type=str, default=os.getenv("OPENAI_MODEL"), help="Model name for API call.")
    parser.add_argument("--api-timeout", type=int, default=120, help="API request timeout in seconds.")
    parser.add_argument("--api-temperature", type=float, default=0.0, help="Temperature for API generation.")
    parser.add_argument("--api-max-tokens", type=int, default=1200, help="Max output tokens for API generation.")
    parser.add_argument("--request-sleep-seconds", type=float, default=0.0, help="Sleep between API requests.")
    parser.add_argument("--skip-api", action="store_true", help="Skip API calls and save empty counterfactual outputs.")

    args = parser.parse_args()

    rng = random.Random(args.seed)

    traces_json = args.traces_json or resolve_default_traces_json(args.model_name, args.experiment)
    if not traces_json.exists():
        raise FileNotFoundError(f"Traces file not found: {traces_json}")

    loaded_traces = load_trace_list_from_json(traces_json)
    if args.max_traces is not None:
        if args.max_traces <= 0:
            raise ValueError("--max-traces must be > 0")
        loaded_traces = loaded_traces[: args.max_traces]

    output_json = args.output_json or resolve_default_output_json(args.model_name, args.experiment)
    tokenizer_path = args.tokenizer_path or resolve_default_tokenizer_path(args.model_name)
    prompt_experiment_name = get_prompt_experiment_name(args.experiment)

    # exit if output already exists
    if output_json.exists():
        print(f"Output JSON already exists at {output_json}, exiting to avoid overwrite.")
        return

    if not args.skip_api and not args.api_url:
        raise ValueError("API URL is required unless --skip-api is set. Use --api-url or OPENAI_ENDPOINT.")

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    print(f"Loaded traces: {len(loaded_traces)}")
    pairs_out: List[Dict[str, Any]] = []

    for out_id, source_trace in enumerate(loaded_traces):
        (
            prompt_text,
            format_id,
            prompt_metadata,
            source_trace_id,
            source_generated_text,
            source_tokens,
            _source_token_strings,
            _source_prompt_length,
        ) = split_trace_fields(source_trace)

        source_generated_text, _ = truncate_after_first_question_block(source_generated_text)

        cf_prompt_metadata, replacements = build_counterfactual_metadata(
            prompt_experiment_name=prompt_experiment_name,
            format_id=format_id,
            original_prompt_metadata=prompt_metadata,
            rng=rng,
        )

        status = "skipped"
        error_message = None
        generated_text = None
        source_values = extract_numeric_spans_from_text(source_generated_text, tokenizer)

        if not args.skip_api:
            api_prompt = build_generation_prompt(
                prompt_text=prompt_text,
                base_trace=source_generated_text,
                original_prompt_metadata=prompt_metadata,
                counterfactual_prompt_metadata=cf_prompt_metadata,
                replacements=replacements,
            )
            response_text, _raw, error_message = call_api(
                api_url=args.api_url,
                api_key=args.api_key,
                api_model=args.api_model,
                user_prompt=api_prompt,
                timeout_s=args.api_timeout,
                temperature=args.api_temperature,
                max_tokens=args.api_max_tokens,
            )

            if error_message is None:
                parsed = extract_first_json_object(response_text or "")
                if parsed is None:
                    status = "error"
                    error_message = "Could not parse generator JSON output"
                else:
                    generated_text = parsed.get("generated_trace")
                    status = "ok"
            else:
                status = "error"

            if args.request_sleep_seconds > 0:
                time.sleep(args.request_sleep_seconds)

        counterfactual_values = extract_numeric_spans_from_text(generated_text, tokenizer)

        token_ids, _token_strings = tokenize_text(tokenizer, generated_text)
        prompt_token_ids, _prompt_token_strings = tokenize_text(tokenizer, prompt_text)

        pairs_out.append(
            {
                "id": out_id,
                "trace_id": source_trace_id,
                "prompt": prompt_text,
                "prompt_tokens": prompt_token_ids,
                "pair": {
                    "source": {
                        "generated_text": source_generated_text,
                        "tokens": source_tokens,
                        "values": source_values,
                    },
                    "counterfactual": {
                        "generated_text": generated_text,
                        "tokens": token_ids,
                        "values": counterfactual_values,
                    },
                },
            }
        )

        print(f"[{out_id + 1}/{len(loaded_traces)}] trace_id={source_trace_id} status={status} replacements={len(replacements)}")
        if error_message:
            print(f"  error: {error_message}")
        print(f"  source        : {source_generated_text}")
        print(f"  counterfactual: {generated_text}")

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pairs": pairs_out,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("=" * 80)
    print("PAIRED COUNTERFACTUAL GENERATION COMPLETE")
    print("=" * 80)
    print(f"Experiment: {args.experiment}")
    print(f"Input fixed traces JSON: {traces_json}")
    print(f"Generated pairs: {len(pairs_out)}")
    print(f"API calls enabled: {not args.skip_api}")
    print(f"Output JSON: {output_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
