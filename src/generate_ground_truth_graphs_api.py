#!/usr/bin/env python3
"""
Generate LLM-judged "ground-truth" value graphs from aligned pair traces.

This script:
1) Reads aligned_pairs.json (same source used by graph construction).
2) Labels numeric values as v0, v1, ... (same reversed indexing convention).
3) Calls an API model to infer directed parent->child edges over those values.
4) Saves graph.json / graph.dot (and optional rendered graph image) per pair.

Output graph schema intentionally mirrors construct_graph.py output to simplify
downstream comparison against candidate graphs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib import error, request

try:
    from dotenv import load_dotenv as _load_dotenv
except Exception:  # pragma: no cover - optional dependency
    _load_dotenv = None

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workspace_paths import resolve_auto_traces_root

def load_env_file_fallback(path: Path) -> bool:
    if not path.exists():
        return False
    loaded_any = False
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        if key not in os.environ:
            os.environ[key] = value
            loaded_any = True
    return loaded_any


def load_env_vars() -> None:
    if _load_dotenv is not None:
        _load_dotenv()
        return
    # Minimal fallback when python-dotenv is unavailable.
    load_env_file_fallback(Path.cwd() / ".env")
    load_env_file_fallback(Path(__file__).resolve().parent / ".env")


load_env_vars()


EDGE_TEXT_RE = re.compile(r"\b(v\d+)\s*[-=]>\s*(v\d+)\b")


@dataclass
class GTNode:
    node_id: str
    value_index: int
    source_index: int
    value_text: str
    normalized_value: Optional[str]
    token_start: Optional[int]
    token_end: Optional[int]
    span_start: Optional[int]
    span_end: Optional[int]
    truncation_token_index: int
    is_prompt_value: Optional[bool]


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


def resolve_default_aligned_pairs_json(model_name: str, experiment: str) -> Path:
    return resolve_auto_traces_root(model_name, experiment) / "aligned_pairs.json"


def resolve_default_output_dir(model_name: str, experiment: str) -> Path:
    return resolve_auto_traces_root(model_name, experiment) / "graphs_ground_truth_api"


def load_pairs(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        pairs = payload.get("pairs")
        if isinstance(pairs, list):
            return [p for p in pairs if isinstance(p, dict)]
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]

    raise ValueError(f"Unsupported aligned_pairs structure in {path}")


def get_pair_id(pair: Dict[str, Any]) -> str:
    raw = pair.get("pair_id", pair.get("id", ""))
    return str(raw)


def normalize_pair_dir_name(pair_id: str) -> str:
    return pair_id if pair_id.startswith("pair") else f"pair{pair_id}"


def get_source_and_cf(pair: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    nested_pair = pair.get("pair", {}) if isinstance(pair.get("pair"), dict) else {}
    source = nested_pair.get("source", {}) if isinstance(nested_pair.get("source"), dict) else {}
    counterfactual = (
        nested_pair.get("counterfactual", {}) if isinstance(nested_pair.get("counterfactual"), dict) else {}
    )
    return source, counterfactual


def infer_prompt_token_cutoff(pair: Dict[str, Any], source_values: List[Dict[str, Any]]) -> Optional[int]:
    prompt_text = str(pair.get("prompt", "") or "")
    if not prompt_text:
        return None
    prompt_len_chars = len(prompt_text)
    candidates: List[int] = []
    for value_obj in source_values:
        span_start = value_obj.get("span_start")
        token_start = value_obj.get("token_start")
        if isinstance(span_start, int) and isinstance(token_start, int) and span_start >= prompt_len_chars:
            candidates.append(token_start)
    if not candidates:
        return None
    return int(min(candidates))


def compute_excluded_shared_values(
    source_values: List[Dict[str, Any]], counterfactual_values: List[Dict[str, Any]]
) -> Tuple[Set[int], List[Dict[str, Any]]]:
    """
    Return graph value indices that are shared source/counterfactual constants.

    Value indices follow reversed convention:
      source_values[src_idx] -> graph value_index = n_source_values - 1 - src_idx
    """
    excluded_indices: Set[int] = set()
    metadata: List[Dict[str, Any]] = []
    source_count = len(source_values)
    for src_idx, (src, cf) in enumerate(zip(source_values, counterfactual_values)):
        if not isinstance(src, dict) or not isinstance(cf, dict):
            continue
        src_norm = normalize_number_string(src.get("normalized_value", src.get("value_text")))
        cf_norm = normalize_number_string(cf.get("normalized_value", cf.get("value_text")))
        if not src_norm or src_norm != cf_norm:
            continue
        value_index = source_count - 1 - src_idx
        if value_index < 0:
            continue
        excluded_indices.add(value_index)
        metadata.append(
            {
                "value_index": value_index,
                "value_text": src.get("value_text"),
                "normalized_value": src_norm,
                "span_start": src.get("span_start"),
                "span_end": src.get("span_end"),
                "reason": "shared_source_counterfactual",
            }
        )
    return excluded_indices, metadata


def build_nodes(
    pair: Dict[str, Any],
    source_values: List[Dict[str, Any]],
    exclude_shared_indices: Set[int],
) -> List[GTNode]:
    prompt_cutoff = infer_prompt_token_cutoff(pair, source_values)
    source_count = len(source_values)
    out: List[GTNode] = []
    for src_idx, value_obj in enumerate(source_values):
        if not isinstance(value_obj, dict):
            continue
        value_index = source_count - 1 - src_idx
        if value_index in exclude_shared_indices:
            continue

        token_start = value_obj.get("token_start") if isinstance(value_obj.get("token_start"), int) else None
        token_end = value_obj.get("token_end") if isinstance(value_obj.get("token_end"), int) else None
        span_start = value_obj.get("span_start") if isinstance(value_obj.get("span_start"), int) else None
        span_end = value_obj.get("span_end") if isinstance(value_obj.get("span_end"), int) else None

        if token_start is not None:
            trunc = int(token_start)
        elif token_end is not None and token_end > 0:
            trunc = int(token_end - 1)
        else:
            trunc = int(src_idx)

        is_prompt_value: Optional[bool] = None
        if prompt_cutoff is not None and token_start is not None:
            is_prompt_value = token_start < prompt_cutoff

        value_text = str(value_obj.get("value_text", "") or "")
        norm = normalize_number_string(value_obj.get("normalized_value", value_text))

        out.append(
            GTNode(
                node_id=f"v{value_index}",
                value_index=value_index,
                source_index=src_idx,
                value_text=value_text,
                normalized_value=norm,
                token_start=token_start,
                token_end=token_end,
                span_start=span_start,
                span_end=span_end,
                truncation_token_index=trunc,
                is_prompt_value=is_prompt_value,
            )
        )
    return sorted(out, key=lambda n: (n.truncation_token_index, n.value_index))


def annotate_trace_with_node_labels(text: str, nodes: List[GTNode]) -> str:
    if not text:
        return text

    inserts: List[Tuple[int, str]] = []
    for n in nodes:
        if n.span_end is None:
            continue
        if not (0 <= n.span_end <= len(text)):
            continue
        inserts.append((n.span_end, f"[{n.node_id}]"))

    if not inserts:
        return text

    # Insert from right to left.
    out = text
    for pos, label in sorted(inserts, key=lambda x: x[0], reverse=True):
        out = out[:pos] + label + out[pos:]
    return out


def build_node_table(nodes: List[GTNode]) -> str:
    lines = [
        "node_id | value_text | normalized_value | token_span | char_span | trunc_idx",
        "------- | ---------- | ---------------- | ---------- | --------- | ---------",
    ]
    for n in nodes:
        token_span = (
            f"{n.token_start}:{n.token_end}"
            if n.token_start is not None and n.token_end is not None
            else "NA"
        )
        char_span = (
            f"{n.span_start}:{n.span_end}" if n.span_start is not None and n.span_end is not None else "NA"
        )
        lines.append(
            f"{n.node_id} | {n.value_text} | {n.normalized_value} | {token_span} | {char_span} | "
            f"{n.truncation_token_index}"
        )
    return "\n".join(lines)


def build_graph_prompt(
    pair_id: str,
    prompt_text: str,
    source_text: str,
    annotated_text: str,
    nodes: List[GTNode],
    prompt_token_cutoff: Optional[int],
) -> str:
    node_ids = [n.node_id for n in nodes]
    node_table = build_node_table(nodes)
    prompt_cutoff_text = "None" if prompt_token_cutoff is None else str(prompt_token_cutoff)

    return (
        "You are an expert annotator building a DIRECTED ACYCLIC COMPUTATION GRAPH over numeric values in a trace.\n"
        "Each node is a labeled value (v0, v1, ...). Use ONLY provided nodes.\n\n"
        "Edge semantics:\n"
        "- source -> target means source value is directly used to compute target.\n"
        "- Parents must occur earlier in token order than children.\n"
        "- No self-loops, no cycles.\n"
        "- The only nodes without parents should values in the prompt (Before \"Answer (step-by-step):\") --- all other values trace back to values in the prompt. \n"
        "- The graph should be fully connected -- no isolated nodes. \n\n"
        "Output STRICT JSON only:\n"
        "{\n"
        '  "root": "v0",\n'
        '  "edges": [\n'
        '    {"source": "vA", "target": "vB"}\n'
        "  ]\n"
        "}\n"
        f"PAIR_ID: {pair_id}\n"
        f"ALLOWED_NODE_IDS: {node_ids}\n\n"
        f"NODE_TABLE:\n{node_table}\n\n"
        f"SOURCE_TRACE_WITH_LABELS:\n{annotated_text}\n\n"
        f"SOURCE_TRACE_RAW:\n{source_text}\n"
    )


def call_chat_completion_api(
    api_url: str,
    api_key: Optional[str],
    api_model: str,
    user_prompt: str,
    timeout_s: int,
    temperature: float,
    max_tokens: int,
    use_json_mode: bool,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    payload: Dict[str, Any] = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": "You return strict JSON for graph annotation tasks."},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if use_json_mode:
        payload["response_format"] = {"type": "json_object"}

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
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def parse_confidence(raw: Any) -> float:
    try:
        val = float(raw)
    except Exception:
        return 1.0
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def parse_edges_from_response(
    parsed: Dict[str, Any],
    allowed_nodes: Set[str],
    node_to_trunc: Dict[str, int],
) -> List[Dict[str, Any]]:
    edge_candidates = parsed.get("edges", [])
    normalized_edges: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def add_edge(source: str, target: str, confidence: float = 1.0, reason: Optional[str] = None) -> None:
        if source not in allowed_nodes or target not in allowed_nodes:
            return
        if source == target:
            return
        # Enforce temporal consistency: parent must appear earlier than child.
        if node_to_trunc.get(source, 10**9) >= node_to_trunc.get(target, -1):
            return
        key = (source, target)
        existing = normalized_edges.get(key)
        if existing is None or confidence > existing["weight"]:
            normalized_edges[key] = {
                "source": source,
                "target": target,
                "weight": float(confidence),
                "reason": reason or "",
            }

    if isinstance(edge_candidates, list):
        for item in edge_candidates:
            if isinstance(item, dict):
                source = item.get("source")
                target = item.get("target")
                if isinstance(source, str) and isinstance(target, str):
                    add_edge(
                        source.strip(),
                        target.strip(),
                        confidence=parse_confidence(item.get("confidence", 1.0)),
                        reason=str(item.get("reason", "") or ""),
                    )
            elif isinstance(item, list) and len(item) >= 2:
                src, dst = item[0], item[1]
                if isinstance(src, str) and isinstance(dst, str):
                    add_edge(src.strip(), dst.strip(), confidence=1.0)
            elif isinstance(item, str):
                for m in EDGE_TEXT_RE.finditer(item):
                    add_edge(m.group(1), m.group(2), confidence=1.0)
    elif isinstance(edge_candidates, str):
        for m in EDGE_TEXT_RE.finditer(edge_candidates):
            add_edge(m.group(1), m.group(2), confidence=1.0)

    # Some models may output top-level "links" instead of "edges".
    links = parsed.get("links")
    if isinstance(links, list):
        for item in links:
            if isinstance(item, dict):
                src = item.get("from") or item.get("source")
                dst = item.get("to") or item.get("target")
                if isinstance(src, str) and isinstance(dst, str):
                    add_edge(src.strip(), dst.strip(), confidence=parse_confidence(item.get("confidence", 1.0)))

    return sorted(normalized_edges.values(), key=lambda e: (e["target"], e["source"]))


def build_node_stats(nodes: List[GTNode], edges: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    parents_of: Dict[str, List[str]] = {}
    for e in edges:
        parents_of.setdefault(e["target"], []).append(e["source"])

    stats: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        stats[n.node_id] = {
            "truncation_token_index": int(n.truncation_token_index),
            "n_parent_candidates": None,
            "n_selected_tokens": None,
            "selected_token_positions": [],
            "selected_token_scores_abs": [],
            "mapped_parents": sorted(parents_of.get(n.node_id, [])),
            "api_ground_truth": True,
            "is_prompt_value": n.is_prompt_value,
        }
    return stats


def to_dot(pair_name: str, graph: Dict[str, Any]) -> str:
    def _esc(x: str) -> str:
        return str(x).replace('"', '\\"').replace("\n", "\\n")

    lines = [f"digraph {pair_name} {{", "  rankdir=RL;"]
    for n in graph.get("nodes", []):
        nid = str(n.get("id", ""))
        trunc = int(n.get("truncation_token_index", 0))
        value_texts = n.get("value_texts", []) if isinstance(n.get("value_texts", []), list) else []
        text = value_texts[0] if value_texts else ""
        label = f"{nid}\\n{text}\\nt={trunc}" if text else f"{nid}\\nt={trunc}"
        lines.append(f'  "{_esc(nid)}" [shape=ellipse,label="{_esc(label)}"];')
    for e in graph.get("edges", []):
        src = str(e.get("source", ""))
        dst = str(e.get("target", ""))
        w = float(e.get("weight", 1.0))
        lines.append(f'  "{_esc(src)}" -> "{_esc(dst)}" [label="{w:.2f}"];')
    lines.append("}")
    return "\n".join(lines)


def maybe_render_graph(dot_file: Path, out_file: Path, layout: str, fmt: str) -> bool:
    if fmt == "none":
        return False
    cmd = ["dot", f"-K{layout}", f"-T{fmt}", str(dot_file), "-o", str(out_file)]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        return False
    return proc.returncode == 0


def pick_root(parsed: Dict[str, Any], nodes: List[GTNode]) -> str:
    allowed = {n.node_id for n in nodes}
    requested = parsed.get("root")
    if isinstance(requested, str) and requested in allowed:
        return requested
    # Keep compatibility with current pipeline: prefer v0 if present.
    if "v0" in allowed:
        return "v0"
    # Fallback: latest truncation node.
    if nodes:
        return max(nodes, key=lambda n: n.truncation_token_index).node_id
    return "v0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate API-based ground-truth value graphs from aligned pairs.")
    p.add_argument("--model-name", type=str, default="Qwen2.5-32B", help="Trace model folder name.")
    p.add_argument("--experiment", type=str, default="velocity_from_ke", help="Experiment name.")
    p.add_argument("--aligned-pairs-json", type=Path, default=None, help="Path to aligned_pairs.json.")
    p.add_argument("--output-dir", type=Path, default=None, help="Output graph root directory.")
    p.add_argument("--pair-id", type=str, default=None, help="Optional single pair id (e.g., 0 or pair0).")
    p.add_argument("--max-pairs", type=int, default=0, help="Optional cap on processed pairs (0 = all).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing pair outputs.")
    p.add_argument(
        "--exclude-shared-values",
        action="store_true",
        help="Exclude source/counterfactual-shared values (constants/step numbers).",
    )

    p.add_argument("--api-url", type=str, default=os.getenv("OPENAI_ENDPOINT"), help="Chat completion endpoint URL.")
    p.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"), help="Bearer API key.")
    p.add_argument("--api-model", type=str, default=os.getenv("OPENAI_MODEL"), help="Model name for API call.")
    p.add_argument("--api-timeout", type=int, default=120, help="API request timeout in seconds.")
    p.add_argument("--api-temperature", type=float, default=0.0, help="Sampling temperature.")
    p.add_argument("--api-max-tokens", type=int, default=2000, help="Max output tokens.")
    p.add_argument("--request-sleep-seconds", type=float, default=0.0, help="Sleep between API requests.")
    p.add_argument("--use-json-mode", action="store_true", help="Use response_format={type:json_object}.")
    p.add_argument("--skip-api", action="store_true", help="Skip API call and write empty-edge graphs.")

    p.add_argument("--render", choices=["none", "png", "svg"], default="none", help="Render graph image format.")
    p.add_argument("--layout", choices=["dot", "neato", "fdp", "sfdp", "twopi", "circo"], default="dot")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    aligned_pairs_json = args.aligned_pairs_json or resolve_default_aligned_pairs_json(
        args.model_name, args.experiment
    )
    output_dir = args.output_dir or resolve_default_output_dir(args.model_name, args.experiment)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not aligned_pairs_json.exists():
        raise FileNotFoundError(f"aligned_pairs.json not found: {aligned_pairs_json}")

    if not args.skip_api:
        if not args.api_url:
            raise ValueError("API URL is required unless --skip-api is set.")
        if not args.api_model:
            raise ValueError("API model is required unless --skip-api is set.")

    pairs = load_pairs(aligned_pairs_json)
    if args.pair_id is not None:
        wanted = normalize_pair_dir_name(str(args.pair_id))
        pairs = [p for p in pairs if normalize_pair_dir_name(get_pair_id(p)) == wanted]
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    if not pairs:
        print("No pairs selected. Nothing to do.")
        return

    summary: Dict[str, Any] = {
        "schema_version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": {
            "model_name": args.model_name,
            "experiment": args.experiment,
            "aligned_pairs_json": str(aligned_pairs_json),
            "output_dir": str(output_dir),
            "exclude_shared_values": bool(args.exclude_shared_values),
            "api_model": args.api_model,
            "render": args.render,
            "layout": args.layout,
        },
        "pairs": [],
    }

    for idx, pair in enumerate(pairs, start=1):
        pair_id = get_pair_id(pair)
        pair_name = normalize_pair_dir_name(pair_id)
        pair_out_dir = output_dir / pair_name
        graph_json_path = pair_out_dir / "graph.json"
        graph_dot_path = pair_out_dir / "graph.dot"
        raw_response_path = pair_out_dir / "api_response.json"

        if graph_json_path.exists() and not args.overwrite:
            print(f"[{idx}/{len(pairs)}] {pair_name}: exists, skipping")
            summary["pairs"].append(
                {
                    "pair": pair_name,
                    "status": "skipped_exists",
                    "graph_json": str(graph_json_path),
                    "graph_dot": str(graph_dot_path) if graph_dot_path.exists() else None,
                }
            )
            continue

        source, counterfactual = get_source_and_cf(pair)
        source_text = str(source.get("generated_text", "") or "")
        prompt_text = str(pair.get("prompt", "") or "")
        source_values = source.get("values", []) if isinstance(source.get("values"), list) else []
        cf_values = counterfactual.get("values", []) if isinstance(counterfactual.get("values"), list) else []
        prompt_token_cutoff = infer_prompt_token_cutoff(pair, source_values)

        excluded_indices: Set[int] = set()
        excluded_metadata: List[Dict[str, Any]] = []
        if args.exclude_shared_values:
            excluded_indices, excluded_metadata = compute_excluded_shared_values(source_values, cf_values)

        nodes = build_nodes(pair, source_values, excluded_indices)
        if not nodes:
            print(f"[{idx}/{len(pairs)}] {pair_name}: no nodes after filtering, writing empty graph")
            graph = {
                "root": "v0",
                "nodes": [],
                "edges": [],
                "node_stats": {},
                "prompt_token_cutoff": prompt_token_cutoff,
                "excluded_shared_values": excluded_metadata,
                "ground_truth_metadata": {
                    "pair_id": pair_id,
                    "status": "empty_nodes",
                    "api_model": args.api_model,
                },
            }
            pair_out_dir.mkdir(parents=True, exist_ok=True)
            with open(graph_json_path, "w") as f:
                json.dump(graph, f, indent=2)
            with open(graph_dot_path, "w") as f:
                f.write(to_dot(pair_name, graph))
            summary["pairs"].append(
                {
                    "pair": pair_name,
                    "status": "empty_nodes",
                    "n_nodes": 0,
                    "n_edges": 0,
                    "graph_json": str(graph_json_path),
                    "graph_dot": str(graph_dot_path),
                    "graph_render": None,
                }
            )
            continue

        annotated_text = annotate_trace_with_node_labels(source_text, nodes)
        user_prompt = build_graph_prompt(
            pair_id=pair_name,
            prompt_text=prompt_text,
            source_text=source_text,
            annotated_text=annotated_text,
            nodes=nodes,
            prompt_token_cutoff=prompt_token_cutoff,
        )

        parsed_response: Dict[str, Any] = {}
        api_error: Optional[str] = None
        response_text: Optional[str] = None
        raw_response: Optional[Dict[str, Any]] = None

        if not args.skip_api:
            response_text, raw_response, api_error = call_chat_completion_api(
                api_url=args.api_url,
                api_key=args.api_key,
                api_model=args.api_model,
                user_prompt=user_prompt,
                timeout_s=args.api_timeout,
                temperature=args.api_temperature,
                max_tokens=args.api_max_tokens,
                use_json_mode=args.use_json_mode,
            )
            if api_error is None:
                parsed = extract_first_json_object(response_text or "")
                if isinstance(parsed, dict):
                    parsed_response = parsed
                else:
                    api_error = "Could not parse JSON object from API response."
            if args.request_sleep_seconds > 0:
                time.sleep(args.request_sleep_seconds)
        else:
            parsed_response = {"root": "v0", "edges": []}

        allowed_nodes = {n.node_id for n in nodes}
        node_to_trunc = {n.node_id: n.truncation_token_index for n in nodes}
        root = pick_root(parsed_response, nodes)
        edges = parse_edges_from_response(parsed_response, allowed_nodes, node_to_trunc)
        node_stats = build_node_stats(nodes, edges)

        graph_nodes = [
            {
                "id": n.node_id,
                "label": n.node_id,
                "truncation_token_index": int(n.truncation_token_index),
                "value_index": int(n.value_index),
                "value_indices": [int(n.value_index)],
                "value_texts": [n.value_text] if n.value_text else [],
            }
            for n in sorted(nodes, key=lambda x: x.truncation_token_index)
        ]

        graph = {
            "root": root,
            "nodes": graph_nodes,
            "edges": edges,
            "node_stats": node_stats,
            "prompt_token_cutoff": prompt_token_cutoff,
            "excluded_shared_values": excluded_metadata,
            "ground_truth_metadata": {
                "pair_id": pair_id,
                "api_model": args.api_model,
                "api_error": api_error,
                "n_allowed_nodes": len(allowed_nodes),
                "response_text_preview": (response_text[:500] if response_text else None),
            },
        }

        pair_out_dir.mkdir(parents=True, exist_ok=True)
        with open(graph_json_path, "w") as f:
            json.dump(graph, f, indent=2)
        with open(graph_dot_path, "w") as f:
            f.write(to_dot(pair_name, graph))

        rendered_path: Optional[Path] = None
        if args.render != "none":
            rendered_path = pair_out_dir / f"graph.{args.render}"
            ok = maybe_render_graph(graph_dot_path, rendered_path, args.layout, args.render)
            if not ok:
                rendered_path = None

        # Save rich API debug payload.
        with open(raw_response_path, "w") as f:
            json.dump(
                {
                    "pair": pair_name,
                    "api_model": args.api_model,
                    "api_error": api_error,
                    "request_prompt": user_prompt,
                    "response_text": response_text,
                    "raw_response": raw_response,
                    "parsed_response": parsed_response,
                },
                f,
                indent=2,
            )

        status = "ok" if api_error is None else "api_error"
        print(
            f"[{idx}/{len(pairs)}] {pair_name}: status={status} nodes={len(graph_nodes)} "
            f"edges={len(edges)}"
        )
        if api_error:
            print(f"  api_error: {api_error}")

        summary["pairs"].append(
            {
                "pair": pair_name,
                "status": status,
                "n_nodes": len(graph_nodes),
                "n_edges": len(edges),
                "graph_json": str(graph_json_path),
                "graph_dot": str(graph_dot_path),
                "graph_render": str(rendered_path) if rendered_path else None,
                "api_response_json": str(raw_response_path),
            }
        )

    summary_path = output_dir / "graph_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
