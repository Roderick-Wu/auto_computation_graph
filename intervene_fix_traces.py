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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

from dotenv import load_dotenv
from transformers import AutoTokenizer


load_dotenv()
REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_default_input_json(model_name: str, experiment: str) -> Path:
    scratch_root = Path.home() / "links" / "scratch"
    if not scratch_root.exists():
        scratch_root = Path.home() / "scratch"
    return scratch_root / "reasoning_traces" / model_name / experiment / "traces.json"


def resolve_default_output_json(model_name: str, experiment: str) -> Path:
    scratch_root = Path.home() / "links" / "scratch"
    if not scratch_root.exists():
        scratch_root = Path.home() / "scratch"
    return scratch_root / "reasoning_traces" / model_name / experiment / "fixed_traces.json"


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
        "2) If incorrect at any step, correct the trace by patching in the valid intermediate values and final answer.\n"
        "3) If correct, return the original trace unchanged.\n\n"
        "Output format requirements (strict):\n"
        "- Return ONLY a JSON object, no markdown.\n"
        "- JSON keys: is_correct (boolean), corrected_trace (string), notes (string).\n"
        "- corrected_trace must be a single plain text block that starts with the question and includes step-by-step reasoning ending with 'The answer is ...'.\n\n"
        f"PROMPT_TEXT:\n{prompt_text}\n\n"
        f"PROMPT_METADATA:\n{json.dumps(prompt_metadata, ensure_ascii=True, sort_keys=True)}\n\n"
        f"EXISTING_TRACE:\n{existing_block}\n"
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

    if not args.skip_api and not args.api_url:
        raise ValueError("API URL is required unless --skip-api is set. Use --api-url or OPENAI_ENDPOINT.")

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    print(f"Loaded traces: {len(loaded_traces)}")

    traces_out: List[Dict[str, Any]] = []

    for out_id, source_trace in enumerate(loaded_traces):
        prompt_text, format_id, prompt_metadata, source_trace_id, existing_text = split_trace_fields(source_trace)

        status = "skipped"
        error_message = None
        is_correct = None
        notes = None
        corrected_text = existing_text

        if not args.skip_api:
            verify_prompt = build_verify_fix_prompt(
                prompt_text=prompt_text,
                prompt_metadata=prompt_metadata,
                existing_trace=existing_text,
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
                    is_correct = bool(parsed.get("is_correct"))
                    corrected_text = parsed.get("corrected_trace") or existing_text
                    notes = parsed.get("notes")
                    status = "ok"
            else:
                status = "error"

            if args.request_sleep_seconds > 0:
                time.sleep(args.request_sleep_seconds)

        token_ids, token_strings = tokenize_text(tokenizer, corrected_text)
        prompt_length = len(tokenizer(prompt_text, return_tensors=None)["input_ids"]) if prompt_text else 0

        traces_out.append(
            {
                "id": out_id,
                "source_trace_id": source_trace_id,
                "format_id": format_id,
                "prompt": prompt_text,
                "prompt_metadata": prompt_metadata,
                "original_generated_text": existing_text,
                "generated_text": corrected_text,
                "tokens": token_ids,
                "token_strings": token_strings,
                "prompt_length": prompt_length,
                "verification": {
                    "status": status,
                    "is_correct": is_correct,
                    "notes": notes,
                    "error": error_message,
                    "api_model": None if args.skip_api else args.api_model,
                },
            }
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "verify_and_fix",
        "input": {
            "experiment": args.experiment,
            "input_json": str(input_json),
            "n_loaded_traces": len(loaded_traces),
        },
        "api": {
            "skip_api": args.skip_api,
            "api_url": args.api_url,
            "api_model": args.api_model,
            "api_temperature": args.api_temperature,
            "api_max_tokens": args.api_max_tokens,
            "api_timeout": args.api_timeout,
        },
        "n_traces": len(traces_out),
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
