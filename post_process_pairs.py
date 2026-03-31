"""
Post-process paired traces to improve source/counterfactual token-length alignment.

For Qwen tokenizers, digits are typically tokenized per character, so matching
numeric string lengths helps match token lengths when only numeric values differ.

Input: paired_traces.json from intervene_generate_pairs.py
Output: aligned_pairs.json with updated counterfactual text/tokens and diagnostics.
"""

import argparse
import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer


NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w]|\.\d)")


def resolve_default_input_json(model_name: str, experiment: str) -> Path:
    scratch_root = Path.home() / "links" / "scratch"
    if not scratch_root.exists():
        scratch_root = Path.home() / "scratch"
    return scratch_root / "traces" / model_name / experiment / "paired_traces.json"


def resolve_default_output_json(model_name: str, experiment: str) -> Path:
    scratch_root = Path.home() / "links" / "scratch"
    if not scratch_root.exists():
        scratch_root = Path.home() / "scratch"
    return scratch_root / "traces" / model_name / experiment / "aligned_pairs.json"


def resolve_default_tokenizer_path(repo_root: Path, model_name: str) -> Path:
    return repo_root / "models" / model_name


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


def extract_number_spans(text: Optional[str]) -> List[Tuple[int, int, str]]:
    if not text:
        return []
    out: List[Tuple[int, int, str]] = []
    for m in NUMBER_PATTERN.finditer(text):
        start, end = m.span()
        out.append((start, end, m.group(0)))
    return out


def try_get_token_offsets(tokenizer: Any, text: str) -> Optional[List[Tuple[int, int]]]:
    try:
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    except Exception:
        return None

    offsets = encoded.get("offset_mapping") if isinstance(encoded, dict) else None
    if isinstance(offsets, list):
        return [(int(start), int(end)) for start, end in offsets]

    # Fallback when offset mapping is unavailable: reconstruct offsets from decoded tokens.
    try:
        token_ids = tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"]
        token_strings = [tokenizer.decode([tid]) for tid in token_ids]
    except Exception:
        return None

    out: List[Tuple[int, int]] = []
    cursor = 0
    for tok in token_strings:
        if tok == "":
            out.append((cursor, cursor))
            continue

        if text.startswith(tok, cursor):
            start = cursor
            end = start + len(tok)
            out.append((start, end))
            cursor = end
            continue

        found = text.find(tok, cursor)
        if found == -1:
            # Approximate fallback for decode mismatches: advance by token text length.
            start = cursor
            end = min(len(text), start + len(tok))
            out.append((start, end))
            cursor = end
            continue
        start = found
        end = start + len(tok)
        out.append((start, end))
        cursor = end

    return out


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

    if token_start is None:
        for idx, (tok_start, _tok_end) in enumerate(offsets):
            if tok_start >= span_start:
                token_start = idx
                token_end = idx + 1
                break
        if token_start is None and offsets:
            token_start = len(offsets) - 1
            token_end = len(offsets)

    if token_start is not None and token_end is None:
        token_end = token_start + 1

    return token_start, token_end


def extract_numeric_spans_with_tokens(text: Optional[str], tokenizer: Any) -> List[Dict[str, Any]]:
    if not text:
        return []

    offsets = try_get_token_offsets(tokenizer, text)
    out: List[Dict[str, Any]] = []
    for start, end, value_text in extract_number_spans(text):
        token_start, token_end = char_span_to_token_span(offsets, start, end)
        out.append(
            {
                "value_text": value_text,
                "normalized_value": normalize_number_string(value_text),
                "span_start": start,
                "span_end": end,
                "token_start": token_start,
                "token_end": token_end,
            }
        )
    return out


def build_matched_values_metadata(source_values: List[Dict[str, Any]], counterfactual_values: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    for idx, (src, cf) in enumerate(zip(source_values, counterfactual_values)):
        matched.append(
            {
                "position": idx,
                "source": {
                    "value_text": src.get("value_text"),
                    "normalized_value": src.get("normalized_value"),
                    "span_start": src.get("span_start"),
                    "span_end": src.get("span_end"),
                    "token_start": src.get("token_start"),
                    "token_end": src.get("token_end"),
                },
                "counterfactual": {
                    "value_text": cf.get("value_text"),
                    "normalized_value": cf.get("normalized_value"),
                    "span_start": cf.get("span_start"),
                    "span_end": cf.get("span_end"),
                    "token_start": cf.get("token_start"),
                    "token_end": cf.get("token_end"),
                },
            }
        )
    return matched


def parse_float(value: str) -> Optional[float]:
    normalized = normalize_number_string(value)
    if normalized is None:
        return None
    try:
        return float(normalized)
    except Exception:
        return None


def pad_value_to_target_length(value_text: str, other_value_text: str, target_len: int) -> str:
    """Pad a value to target length by appending decimal zeros (never prepend)."""
    if len(value_text) >= target_len:
        return value_text

    missing = target_len - len(value_text)
    if "." in value_text:
        return value_text + ("0" * missing)

    out = value_text + "."
    zeros_needed = target_len - len(out)
    if zeros_needed < 0:
        zeros_needed = 0
    return out + ("0" * zeros_needed)


def rewrite_pair_by_matched_lengths(source_text: str, cf_text: str) -> Tuple[str, str, Dict[str, Any]]:
    source_nums = extract_number_spans(source_text)
    cf_nums = extract_number_spans(cf_text)

    diagnostics: Dict[str, Any] = {
        "source_numeric_count": len(source_nums),
        "counterfactual_numeric_count": len(cf_nums),
        "aligned_positions": 0,
        "source_changed_values": 0,
        "counterfactual_changed_values": 0,
        "differing_values": [],
    }

    if not source_nums or not cf_nums:
        diagnostics["status"] = "no_numeric_values"
        return source_text, cf_text, diagnostics

    if len(source_nums) != len(cf_nums):
        diagnostics["status"] = "count_mismatch"
        return source_text, cf_text, diagnostics

    src_pieces = []
    src_cursor = 0
    cf_pieces = []
    cf_cursor = 0
    src_changed = 0
    cf_changed = 0
    differing_values: List[Dict[str, Any]] = []

    for idx, ((src_start, src_end, src_num), (cf_start, cf_end, cf_num)) in enumerate(zip(source_nums, cf_nums)):
        # For integer pairs that differ by exactly one digit in length,
        # make both decimal first, then pad with decimal zeros to equal length.
        src_for_pad = src_num
        cf_for_pad = cf_num
        if "." not in src_num and "." not in cf_num and abs(len(src_num) - len(cf_num)) == 1:
            src_for_pad = src_num + ".0"
            cf_for_pad = cf_num + ".0"

        target_len = max(len(src_for_pad), len(cf_for_pad))
        new_src_num = pad_value_to_target_length(src_for_pad, cf_for_pad, target_len)
        new_cf_num = pad_value_to_target_length(cf_for_pad, src_for_pad, target_len)

        src_pieces.append(source_text[src_cursor:src_start])
        src_pieces.append(new_src_num)
        src_cursor = src_end

        cf_pieces.append(cf_text[cf_cursor:cf_start])
        cf_pieces.append(new_cf_num)
        cf_cursor = cf_end

        diagnostics["aligned_positions"] += 1
        if src_num != cf_num:
            differing_values.append(
                {
                    "position": idx,
                    "source_original": src_num,
                    "source_rewritten": new_src_num,
                    "counterfactual_original": cf_num,
                    "counterfactual_rewritten": new_cf_num,
                }
            )
        if new_src_num != src_num:
            src_changed += 1
        if new_cf_num != cf_num:
            cf_changed += 1

    src_pieces.append(source_text[src_cursor:])
    cf_pieces.append(cf_text[cf_cursor:])
    new_source_text = "".join(src_pieces)
    new_cf_text = "".join(cf_pieces)

    diagnostics["source_changed_values"] = src_changed
    diagnostics["counterfactual_changed_values"] = cf_changed
    diagnostics["differing_values"] = differing_values
    diagnostics["status"] = "ok"
    return new_source_text, new_cf_text, diagnostics


def tokenize_text(tokenizer: Any, text: Optional[str]) -> Tuple[List[int], List[str]]:
    if not text:
        return [], []
    token_ids = tokenizer(text, return_tensors=None)["input_ids"]
    token_strings = [tokenizer.decode([tid]) for tid in token_ids]
    return token_ids, token_strings


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process paired traces for token-length alignment.")
    parser.add_argument("--model-name", type=str, default="Qwen2.5-72B")
    parser.add_argument("--experiment", type=str, default="velocity")
    parser.add_argument("--input-json", type=Path, default=None, help="Input paired_traces.json")
    parser.add_argument("--output-json", type=Path, default=None, help="Output aligned_pairs.json")
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    input_json = args.input_json or resolve_default_input_json(args.model_name, args.experiment)
    output_json = args.output_json or resolve_default_output_json(args.model_name, args.experiment)
    tokenizer_path = args.tokenizer_path or resolve_default_tokenizer_path(repo_root, args.model_name)

    if not input_json.exists():
        raise FileNotFoundError(f"Input paired traces not found: {input_json}")

    with open(input_json, "r") as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or not isinstance(payload.get("pairs"), list):
        raise ValueError("Expected paired traces JSON payload containing top-level 'pairs' list")

    pairs = payload["pairs"]
    if args.max_pairs is not None:
        if args.max_pairs <= 0:
            raise ValueError("--max-pairs must be > 0")
        pairs = pairs[: args.max_pairs]

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    out_pairs: List[Dict[str, Any]] = []
    equal_token_count = 0

    for pair in pairs:
        source_text = pair.get("pair", {}).get("source", {}).get("generated_text") or ""
        cf_text = pair.get("pair", {}).get("counterfactual", {}).get("generated_text") or ""

        new_source_text, new_cf_text, diagnostics = rewrite_pair_by_matched_lengths(source_text, cf_text)

        src_token_ids, src_token_strings = tokenize_text(tokenizer, new_source_text)
        cf_token_ids, cf_token_strings = tokenize_text(tokenizer, new_cf_text)
        source_values = extract_numeric_spans_with_tokens(new_source_text, tokenizer)
        counterfactual_values = extract_numeric_spans_with_tokens(new_cf_text, tokenizer)
        matched_values = build_matched_values_metadata(source_values, counterfactual_values)
        diagnostics["matched_values"] = matched_values
        diagnostics["matched_values_count"] = len(matched_values)
        diagnostics["matched_values_complete"] = len(source_values) == len(counterfactual_values)

        token_lengths_equal = len(src_token_ids) == len(cf_token_ids)
        if token_lengths_equal:
            equal_token_count += 1

        pair_copy = dict(pair)
        pair_copy.setdefault("pair", {}).setdefault("source", {})
        pair_copy["pair"]["source"]["generated_text"] = new_source_text
        pair_copy["pair"]["source"]["tokens"] = src_token_ids
        pair_copy["pair"]["source"]["token_strings"] = src_token_strings
        pair_copy["pair"]["source"]["prompt_length"] = len(src_token_ids)
        pair_copy["pair"]["source"]["values"] = source_values

        pair_copy.setdefault("pair", {}).setdefault("counterfactual", {})
        pair_copy["pair"]["counterfactual"]["generated_text"] = new_cf_text
        pair_copy["pair"]["counterfactual"]["tokens"] = cf_token_ids
        pair_copy["pair"]["counterfactual"]["token_strings"] = cf_token_strings
        pair_copy["pair"]["counterfactual"]["prompt_length"] = len(cf_token_ids)
        pair_copy["pair"]["counterfactual"]["values"] = counterfactual_values

        pair_copy["post_process"] = {
            "numeric_length_alignment": diagnostics,
            "source_token_count": len(src_token_ids),
            "counterfactual_token_count": len(cf_token_ids),
            "token_counts_equal": token_lengths_equal,
        }

        out_pairs.append(pair_copy)

        print(f"Processed pair {pair.get('id', 'unknown')}: ")
        print(f"  Original source : \n{source_text}")
        print(f"  Original CF text: \n{cf_text}")
        print(f"  New source text : \n{new_source_text}")
        print(f"  New CF text     : \n{new_cf_text}")
        print(f"  Source tokens: {len(src_token_ids)}, CF tokens: {len(cf_token_ids)}, Equal token count: {token_lengths_equal}")
        print("  Differing values:")
        if diagnostics.get("differing_values"):
            for item in diagnostics.get("differing_values", []):
                print(
                    "    "
                    f"pos={item.get('position')} "
                    f"src_orig={item.get('source_original')} "
                    f"src_new={item.get('source_rewritten')} "
                    f"cf_orig={item.get('counterfactual_original')} "
                    f"cf_new={item.get('counterfactual_rewritten')}"
                )
        else:
            print("    none")
        print(f"  Diagnostics: {diagnostics}")
        print("-" * 40)
    out_payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "post_process_pairs_token_alignment",
        "input": {
            "experiment": args.experiment,
            "input_json": str(input_json),
            "n_loaded_pairs": len(pairs),
        },
        "output": {
            "n_pairs": len(out_pairs),
            "n_equal_token_count": equal_token_count,
            "equal_token_ratio": (equal_token_count / len(out_pairs)) if out_pairs else 0.0,
        },
        "pairs": out_pairs,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(out_payload, f, indent=2)

    print("=" * 80)
    print("POST-PROCESS PAIRS COMPLETE")
    print("=" * 80)
    print(f"Input JSON: {input_json}")
    print(f"Output JSON: {output_json}")
    print(f"Pairs processed: {len(out_pairs)}")
    print(f"Token-count-equal pairs: {equal_token_count}/{len(out_pairs)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
