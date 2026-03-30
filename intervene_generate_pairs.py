"""
Generate traces via API only from pre-generated traces JSON.

Pipeline role:
1. Load traces.json produced by generate_traces.py
2. For each loaded prompt, query an API to generate a full reasoning trace
3. Tokenize the generated text
4. Save all API outputs to JSON for downstream use

Environment variables (optional):
    COUNTERFACTUAL_API_URL   Full chat completion endpoint URL
    COUNTERFACTUAL_API_KEY   Bearer token
    COUNTERFACTUAL_API_MODEL Model name sent to the API
    OPENAI_API_KEY           Fallback API key if COUNTERFACTUAL_API_KEY is unset

Examples:
    python intervene_generate_pairs.py --experiment velocity
    python intervene_generate_pairs.py --traces-json /path/to/fixed_traces.json
    python intervene_generate_pairs.py --experiment current --output-json /tmp/paired_traces.json
    python intervene_generate_pairs.py --skip-api
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

from dotenv import load_dotenv
from transformers import AutoTokenizer


load_dotenv()
REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_default_traces_json(model_name: str, experiment: str) -> Path:
    scratch_root = Path.home() / "links" / "scratch"
    if not scratch_root.exists():
        scratch_root = Path.home() / "scratch"
    return scratch_root / "traces" / model_name / experiment / "fixed_traces.json"


def load_trace_list_from_json(path: Path) -> List[Dict[str, Any]]:
    """Load trace list from fixed_traces payload (dict containing 'traces')."""
    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and isinstance(payload.get("traces"), list):
        return payload["traces"]

    raise ValueError("Expected fixed_traces JSON payload containing a top-level 'traces' list")


def resolve_default_output_json(model_name: str, experiment: str) -> Path:
    scratch_root = Path.home() / "links" / "scratch"
    if not scratch_root.exists():
        scratch_root = Path.home() / "scratch"
    return scratch_root / "traces" / model_name / experiment / "paired_traces.json"


def resolve_default_tokenizer_path(model_name: str) -> Path:
    return REPO_ROOT / "models" / model_name


def split_trace_fields(trace: Dict[str, Any]) -> Tuple[str, Optional[int], Dict[str, Any], Optional[int], Optional[str], List[int], List[str], Optional[int]]:
    """Extract prompt/metadata and source trace payload from an input trace entry."""
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


def build_generation_prompt(prompt_text: str, prompt_metadata: Dict[str, Any]) -> str:
    return (
        "Rewrite the following question and step-by-step reasoning trace with the newly provided values.\n"
        "Hard requirements:\n"
        "- Output one plain text block only.\n"
        "- Start with the given question text exactly as provided.\n"
        "- Do not include markdown, JSON, or commentary.\n\n"
        f"PROMPT_TEXT:\n{prompt_text}\n\n"
        f"PROMPT_METADATA:\n{json.dumps(prompt_metadata, ensure_ascii=True, sort_keys=True)}\n"
        f""
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
                "content": "You produce precise math and physics reasoning traces.",
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
    parser.add_argument(
        "--traces-json",
        type=Path,
        default=None,
        help="Path to fixed_traces.json produced by intervene_fix_traces.py.",
    )
    parser.add_argument("--max-traces", type=int, default=None, help="Optional cap on number of loaded traces to process.")
    parser.add_argument("--output-json", type=Path, default=None, help="Output path for API traces JSON.")
    parser.add_argument("--tokenizer-path", type=Path, default=None, help="Tokenizer path for tokenizing generated traces.")

    parser.add_argument("--api-url", type=str, default=os.getenv("OPENAI_ENDPOINT"), help="Chat completion endpoint URL.")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"), help="Bearer API key.")
    parser.add_argument("--api-model", type=str, default=os.getenv("OPENAI_MODEL"), help="Model name for API call.")
    parser.add_argument("--api-timeout", type=int, default=120, help="API request timeout in seconds.")
    parser.add_argument("--api-temperature", type=float, default=0.0, help="Temperature for API generation.")
    parser.add_argument("--api-max-tokens", type=int, default=1200, help="Max output tokens for API generation.")
    parser.add_argument("--request-sleep-seconds", type=float, default=0.0, help="Sleep between API requests.")
    parser.add_argument("--skip-api", action="store_true", help="Skip API calls and save empty outputs.")

    args = parser.parse_args()

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

    if not args.skip_api and not args.api_url:
        raise ValueError("API URL is required unless --skip-api is set. Use --api-url or COUNTERFACTUAL_API_URL.")

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
            source_token_strings,
            source_prompt_length,
        ) = split_trace_fields(source_trace)

        status = "skipped"
        error_message = None
        generated_text = None

        if not args.skip_api:
            api_prompt = build_generation_prompt(prompt_text, prompt_metadata)
            generated_text, _raw, error_message = call_api(
                api_url=args.api_url,
                api_key=args.api_key,
                api_model=args.api_model,
                user_prompt=api_prompt,
                timeout_s=args.api_timeout,
                temperature=args.api_temperature,
                max_tokens=args.api_max_tokens,
            )
            status = "ok" if error_message is None else "error"
            if args.request_sleep_seconds > 0:
                time.sleep(args.request_sleep_seconds)

        token_ids, token_strings = tokenize_text(tokenizer, generated_text)
        prompt_length = len(tokenizer(prompt_text, return_tensors=None)["input_ids"]) if prompt_text else 0

        pairs_out.append(
            {
                "id": out_id,
                "source_trace_id": source_trace_id,
                "format_id": format_id,
                "prompt": prompt_text,
                "prompt_metadata": prompt_metadata,
                "pair": {
                    "source": {
                        "generated_text": source_generated_text,
                        "tokens": source_tokens,
                        "token_strings": source_token_strings,
                        "prompt_length": source_prompt_length,
                    },
                    "counterfactual": {
                        "generated_text": generated_text,
                        "tokens": token_ids,
                        "token_strings": token_strings,
                        "prompt_length": prompt_length,
                    },
                },
                "api": {
                    "status": status,
                    "error": error_message,
                    "api_model": None if args.skip_api else args.api_model,
                },
            }
        )

        print(f"\nProcessed trace ID {source_trace_id} -> pair ID {out_id} with API status: {status}")
        print(f"  Source generated text: {source_generated_text if source_generated_text else 'None'}...")
        print(f"  Counterfactual generated text: {generated_text if generated_text else 'None'}...")

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "paired_counterfactual_from_traces_json",
        "input": {
            "experiment": args.experiment,
            "traces_json": str(traces_json),
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
        "n_pairs": len(pairs_out),
        "pairs": pairs_out,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("=" * 80)
    print("PAIRED TRACE GENERATION COMPLETE")
    print("=" * 80)
    print(f"Experiment: {args.experiment}")
    print(f"Input traces JSON: {traces_json}")
    print(f"Generated pairs: {len(pairs_out)}")
    print(f"API calls enabled: {not args.skip_api}")
    print(f"Output JSON: {output_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
