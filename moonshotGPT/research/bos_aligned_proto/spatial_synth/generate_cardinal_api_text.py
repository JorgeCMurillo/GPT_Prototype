#!/usr/bin/env python3
"""Generate GPT API paraphrases for cardinal spatial-relation mixing.

The script deliberately lets code choose each spatial fact and asks the model
only to lexicalize it. This keeps labels auditable while still adding natural
surface variation for north/south and east/west examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from transformers import AutoTokenizer


CARDINAL = ("north", "south", "east", "west")
INVERSE = {"north": "south", "south": "north", "east": "west", "west": "east"}
FORBIDDEN_WORDS = ("left", "right", "above", "below", "front", "behind", "close", "far", "near", "touch")
FORBIDDEN_PHRASES = ("reaches the", "continues to move straight ahead", "move straight ahead")
CONCEPT_BY_REL = {"north": "north/south", "south": "north/south", "east": "east/west", "west": "east/west"}
CARDINAL_V1_FAMILIES = (
    "api_cardinal_inverse",
    "api_cardinal_move_past",
    "api_cardinal_landmark",
    "api_cardinal_route_update",
)
PASS_THROUGH_V2_FAMILIES = (
    "api_cardinal_pass_through_landmark_final",
    "api_cardinal_pass_through_actor_final",
    "api_cardinal_route_first_pass_through",
    "api_cardinal_turn_around_invariance",
    "api_cardinal_static_inverse_anchor",
)
ALL_TEMPLATE_FAMILIES = (*CARDINAL_V1_FAMILIES, *PASS_THROUGH_V2_FAMILIES)
# Controlled variation: Python samples the spatial rule, final relation, and
# one of a small number of template families; GPT-5.2 only paraphrases within
# that family. This avoids a single brittle prompt format while keeping the
# target relation auditable and preventing unconstrained story generation.
PASS_THROUGH_V2_WEIGHTS = (25, 25, 20, 20, 10)

PLACES = [
    "the station",
    "the plaza",
    "the library",
    "the garden",
    "the bridge",
    "the fountain",
    "the market",
    "the courtyard",
    "the museum",
    "the bakery",
    "the school",
    "the clinic",
    "the harbor",
    "the orchard",
    "the campsite",
    "the trail marker",
]
LANDMARKS = [
    "the signpost",
    "the stone marker",
    "the gate",
    "the shed",
    "the bench",
    "the flagpole",
    "the map kiosk",
    "the old well",
    "the bus stop",
    "the footbridge",
    "the pine grove",
    "the clock tower",
]
ACTORS = ["Maya", "Noah", "Ava", "Kai", "Mira", "Omar", "Lena", "Jay", "Sofia", "Eli"]


@dataclass(frozen=True)
class CardinalFact:
    fact_id: str
    subject: str
    reference: str
    relation: str
    inverse_relation: str
    final_relation: str
    movement_direction: str
    concept: str
    template_family: str
    difficulty_label: str


def split_context_completion(text: str) -> tuple[str, str]:
    text = " ".join(str(text).strip().split())
    split_at = text.rfind(". ")
    if split_at < 0:
        return "", text
    return text[: split_at + 1].strip(), text[split_at + 2 :].strip()


def read_api_key(path: Path | None) -> str:
    if path is None:
        raise ValueError("Set OPENAI_API_KEY or pass --api-key-file.")
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    return key


def world_phrase(rel: str, ref: str) -> str:
    return f"{rel} of {ref}"


def weighted_choice(rng: random.Random, values: tuple[str, ...], weights: tuple[int, ...]) -> str:
    return rng.choices(list(values), weights=list(weights), k=1)[0]


def make_fact(rng: random.Random, idx: int, preset: str) -> CardinalFact:
    relation = rng.choice(CARDINAL)
    inverse_relation = INVERSE[relation]
    if preset == "cardinal_v1":
        family = rng.choice(CARDINAL_V1_FAMILIES)
    elif preset == "pass_through_v2":
        family = weighted_choice(rng, PASS_THROUGH_V2_FAMILIES, PASS_THROUGH_V2_WEIGHTS)
    else:
        raise ValueError(f"Unsupported preset={preset!r}")

    if family == "api_cardinal_move_past":
        subject = rng.choice(ACTORS)
        reference = rng.choice(LANDMARKS)
        difficulty = "hard"
    elif family == "api_cardinal_route_update":
        subject, reference = rng.sample(PLACES + LANDMARKS, 2)
        difficulty = "medium"
    elif family == "api_cardinal_landmark":
        subject, reference = rng.sample(PLACES + LANDMARKS, 2)
        difficulty = "medium"
    elif family in {
        "api_cardinal_pass_through_landmark_final",
        "api_cardinal_pass_through_actor_final",
        "api_cardinal_route_first_pass_through",
    }:
        subject = rng.choice(ACTORS)
        reference = rng.choice(LANDMARKS + PLACES)
        difficulty = "hard"
    elif family == "api_cardinal_turn_around_invariance":
        subject = rng.choice(ACTORS)
        reference = rng.choice(LANDMARKS + PLACES)
        difficulty = "medium"
    elif family == "api_cardinal_static_inverse_anchor":
        subject = rng.choice(ACTORS)
        reference = rng.choice(LANDMARKS + PLACES)
        difficulty = "easy"
    else:
        subject, reference = rng.sample(PLACES + LANDMARKS, 2)
        difficulty = "easy"
    if family in {"api_cardinal_pass_through_actor_final", "api_cardinal_turn_around_invariance"}:
        final_relation = relation
    else:
        final_relation = inverse_relation
    return CardinalFact(
        fact_id=f"api_cardinal_{idx:06d}",
        subject=subject,
        reference=reference,
        relation=relation,
        inverse_relation=inverse_relation,
        final_relation=final_relation,
        movement_direction=relation,
        concept=CONCEPT_BY_REL[relation],
        template_family=family,
        difficulty_label=difficulty,
    )


def fact_prompt_row(fact: CardinalFact) -> Dict[str, str]:
    row = asdict(fact)
    if fact.template_family in CARDINAL_V1_FAMILIES:
        row.update(
            {
                "direct_fact": f"{fact.subject} is {world_phrase(fact.relation, fact.reference)}",
                "inverse_fact": f"{fact.reference} is {world_phrase(fact.inverse_relation, fact.subject)}",
            }
        )
        return row

    if fact.template_family == "api_cardinal_pass_through_landmark_final":
        row.update(
            {
                "required_structure": (
                    f"Start with {fact.reference} {world_phrase(fact.relation, fact.subject)}. "
                    f"{fact.subject} moves {fact.movement_direction}, passes {fact.reference}, and stops beyond it. "
                    f"End with {fact.reference} {world_phrase(fact.final_relation, fact.subject)}."
                ),
                "final_fact": f"{fact.reference} is {world_phrase(fact.final_relation, fact.subject)}",
            }
        )
    elif fact.template_family == "api_cardinal_pass_through_actor_final":
        row.update(
            {
                "required_structure": (
                    f"Start with {fact.subject} {world_phrase(fact.inverse_relation, fact.reference)}. "
                    f"{fact.subject} moves {fact.movement_direction}, passes {fact.reference}, and stops beyond it. "
                    f"End with {fact.subject} {world_phrase(fact.relation, fact.reference)}."
                ),
                "final_fact": f"{fact.subject} is {world_phrase(fact.relation, fact.reference)}",
            }
        )
    elif fact.template_family == "api_cardinal_route_first_pass_through":
        row.update(
            {
                "required_structure": (
                    f"Describe {fact.subject}'s route first: {fact.subject} travels {fact.movement_direction}, "
                    f"goes past {fact.reference}, and stops beyond it. "
                    f"Then state that {fact.reference} is {world_phrase(fact.final_relation, fact.subject)}."
                ),
                "final_fact": f"{fact.reference} is {world_phrase(fact.final_relation, fact.subject)}",
            }
        )
    elif fact.template_family == "api_cardinal_turn_around_invariance":
        row.update(
            {
                "required_structure": (
                    f"Start with {fact.reference} {world_phrase(fact.relation, fact.subject)}. "
                    f"{fact.subject} turns around or changes facing direction but does not move location. "
                    f"End with {fact.reference} still {world_phrase(fact.final_relation, fact.subject)}."
                ),
                "final_fact": f"{fact.reference} is still {world_phrase(fact.final_relation, fact.subject)}",
            }
        )
    elif fact.template_family == "api_cardinal_static_inverse_anchor":
        row.update(
            {
                "required_structure": (
                    f"State the reciprocal static fact: if {fact.reference} is "
                    f"{world_phrase(fact.relation, fact.subject)}, then {fact.subject} is "
                    f"{world_phrase(fact.inverse_relation, fact.reference)}."
                ),
                "final_fact": f"{fact.subject} is {world_phrase(fact.inverse_relation, fact.reference)}",
            }
        )
    else:
        raise ValueError(f"Unsupported template family: {fact.template_family}")
    return row


def prompt_for_facts(facts: List[CardinalFact], preset: str) -> str:
    fact_rows = []
    for fact in facts:
        fact_rows.append(fact_prompt_row(fact))

    base = (
        "Generate one short natural training example for each cardinal spatial fact below.\n"
        "Use the exact fact; do not invent or change the spatial relation.\n"
        "Do not imitate benchmark wording. Do not write multiple-choice questions.\n"
        "Allowed relation words: north, south, east, west.\n"
        "Forbidden relation words: left, right, above, below, front, behind, close, far, near, touch.\n"
        "Return exactly one item per fact_id.\n\n"
    )
    if preset == "pass_through_v2":
        base += (
            "This preset targets hard cardinal update rules.\n"
            "Use controlled wording variation: keep the supplied template family and final fact, "
            "but vary ordinary verbs and sentence shape within that family.\n"
            "- For pass-through examples, the actor must move in the same cardinal direction as the initial relation, "
            "pass the landmark/place, and the final relation must flip.\n"
            "- For turn-around examples, the actor changes facing direction but does not move, so the cardinal relation stays the same.\n"
            "- For static anchors, state the reciprocal cardinal relation.\n"
            "Use varied wording such as walks past, goes beyond, passes, crosses the marker, drives past, or stops beyond it.\n"
            "Do not use the exact phrase 'reaches the X and continues to move straight ahead'.\n"
            "Keep each example one or two short sentences.\n\n"
        )
    else:
        base += "Each text must mention both the direct relation and the inverse relation, using the supplied words.\n\n"
    return base + f"Facts:\n{json.dumps(fact_rows, ensure_ascii=False, indent=2)}"


def response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fact_id": {"type": "string"},
                        "text": {"type": "string"},
                        "concept": {"type": "string", "enum": ["north/south", "east/west"]},
                        "template_family": {
                            "type": "string",
                            "enum": [
                                *ALL_TEMPLATE_FAMILIES,
                            ],
                        },
                        "relation": {"type": "string", "enum": list(CARDINAL)},
                        "inverse_relation": {"type": "string", "enum": list(CARDINAL)},
                        "final_relation": {"type": "string", "enum": list(CARDINAL)},
                    },
                    "required": [
                        "fact_id",
                        "text",
                        "concept",
                        "template_family",
                        "relation",
                        "inverse_relation",
                        "final_relation",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def extract_output_text(payload: Dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    chunks: List[str] = []
    for output in payload.get("output", []):
        for content in output.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    if not chunks:
        raise ValueError(f"Could not find text in response keys={sorted(payload)}")
    return "\n".join(chunks)


def parse_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return exc.reason if isinstance(exc.reason, str) else str(exc)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body[:500]
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or payload)[:500]
    return str(payload)[:500]


def parse_retry_after(headers: Any) -> float | None:
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def rate_limit_summary(headers: Any) -> str:
    keys = [
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
    ]
    parts = []
    for key in keys:
        value = headers.get(key)
        if value:
            parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "no rate-limit headers"


def call_responses_api(
    *,
    api_key: str,
    model: str,
    facts: List[CardinalFact],
    preset: str,
    max_output_tokens: int,
    reasoning_effort: str,
    timeout: int,
) -> List[Dict[str, Any]]:
    body: Dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You write concise natural-language training examples for spatial reasoning. "
                    "You must preserve supplied labels exactly and output valid JSON only."
                ),
            },
            {"role": "user", "content": prompt_for_facts(facts, preset)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "cardinal_examples",
                "strict": True,
                "schema": response_schema(),
            }
        },
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort != "auto":
        body["reasoning"] = {"effort": reasoning_effort}

    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        payload = json.loads(handle.read().decode("utf-8"))
    return json.loads(extract_output_text(payload))["items"]


def validate_generated_item(item: Dict[str, Any], fact_by_id: Dict[str, CardinalFact]) -> str:
    fact_id = str(item.get("fact_id", ""))
    if fact_id not in fact_by_id:
        raise ValueError(f"Unknown fact_id returned: {fact_id!r}")
    fact = fact_by_id[fact_id]
    text = " ".join(str(item.get("text", "")).split())
    if not text:
        raise ValueError(f"Empty text for {fact_id}")
    lower = text.lower()
    bad = [word for word in FORBIDDEN_WORDS if re.search(rf"\b{re.escape(word)}\b", lower)]
    if bad:
        raise ValueError(f"{fact_id} used forbidden relation words {bad}: {text}")
    bad_phrases = [phrase for phrase in FORBIDDEN_PHRASES if phrase in lower]
    if bad_phrases:
        raise ValueError(f"{fact_id} used forbidden benchmark-like phrases {bad_phrases}: {text}")
    if fact.template_family == "api_cardinal_turn_around_invariance":
        if fact.relation not in lower:
            raise ValueError(f"{fact_id} missing invariant relation {fact.relation}: {text}")
    elif fact.relation not in lower or fact.inverse_relation not in lower:
        raise ValueError(
            f"{fact_id} missing relation pair {fact.relation}/{fact.inverse_relation}: {text}"
        )
    if item.get("relation") != fact.relation or item.get("inverse_relation") != fact.inverse_relation:
        raise ValueError(f"{fact_id} returned relation metadata does not match fact")
    if item.get("final_relation") != fact.final_relation:
        raise ValueError(f"{fact_id} returned final_relation metadata does not match fact")
    if item.get("concept") != fact.concept or item.get("template_family") != fact.template_family:
        raise ValueError(f"{fact_id} returned concept/template metadata does not match fact")
    return text


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False)) + 1


def load_existing_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def next_fact_index(rows: List[Dict[str, Any]]) -> int:
    max_idx = -1
    for row in rows:
        fact_id = str(row.get("fact_id", ""))
        try:
            max_idx = max(max_idx, int(fact_id.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return max_idx + 1


def write_rows(rows: List[Dict[str, Any]], out_csv: Path, out_jsonl: Path | None) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "example_id",
        "text",
        "context",
        "completion",
        "loss_start_char",
        "difficulty",
        "difficulty_label",
        "domain",
        "source",
        "generation_preset",
        "concept",
        "relation",
        "inverse_relation",
        "final_relation",
        "movement_direction",
        "template_family",
        "fact_id",
        "subject",
        "reference",
        "api_model",
        "token_count",
        "seed",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if out_jsonl is not None:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_row(
    *,
    example_id: int,
    text: str,
    fact: CardinalFact,
    model: str,
    preset: str,
    tok_count: int,
    seed: int,
) -> Dict[str, Any]:
    context, completion = split_context_completion(text)
    return {
        "example_id": example_id,
        "text": text,
        "context": context,
        "completion": completion,
        "loss_start_char": len(context) + (1 if context and completion else 0),
        "difficulty": {"easy": 1, "medium": 2, "hard": 3}[fact.difficulty_label],
        "difficulty_label": fact.difficulty_label,
        "domain": "spatial-relations",
        "source": f"api_cardinal_gpt52_{preset}",
        "generation_preset": preset,
        "concept": fact.concept,
        "relation": fact.relation,
        "inverse_relation": fact.inverse_relation,
        "final_relation": fact.final_relation,
        "movement_direction": fact.movement_direction,
        "template_family": fact.template_family,
        "fact_id": fact.fact_id,
        "subject": fact.subject,
        "reference": fact.reference,
        "api_model": model,
        "token_count": tok_count,
        "seed": seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument(
        "--preset",
        choices=("cardinal_v1", "pass_through_v2"),
        default="cardinal_v1",
        help="cardinal_v1 is the original reciprocal cardinal generator; pass_through_v2 targets pass-through and turn-around cardinal rules.",
    )
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument(
        "--tokenizer-name",
        default=(
            "/home/jorge/tokenPred/moonshotGPT/experiments/"
            "babygpt_fineweb_bin_mbs4_T1024_d1024_h16_L24_tok491520_efftok491520_ws8_gas15_seed42_steps20000/"
            "ckpt_periodic_step0008000"
        ),
    )
    parser.add_argument("--target-tokens", type=int, default=39219, help="Default is about 10% of a 10k run.")
    parser.add_argument("--request-size", type=int, default=16)
    parser.add_argument("--max-requests", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--reasoning-effort", choices=("auto", "none", "low", "medium", "high", "xhigh"), default="none")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-api-attempts", type=int, default=8)
    parser.add_argument("--rate-limit-sleep", type=float, default=60.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/home/jorge/tokenPred/moonshotGPT/runs/research/bos_aligned_proto/spatial_synth/"
            "cardinal_api_gpt52_10pct_tokens39219_seed42.csv"
        ),
    )
    parser.add_argument("--jsonl-out", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Continue from an existing output CSV if present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.request_size <= 0:
        raise ValueError("--request-size must be positive")
    rng = random.Random(args.seed)
    api_key = os.environ.get("OPENAI_API_KEY") or read_api_key(args.api_key_file)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    rows = load_existing_rows(args.out) if args.resume else []
    total_tokens = sum(int(row.get("token_count", 0) or 0) for row in rows)
    next_example_id = len(rows)
    next_fact_idx = next_fact_index(rows)

    print(
        f"Generating API cardinal text with model={args.model}, preset={args.preset}, target_tokens={args.target_tokens}, "
        f"existing_rows={len(rows)}, existing_tokens={total_tokens}",
        flush=True,
    )
    requests_made = 0
    while total_tokens < args.target_tokens and requests_made < args.max_requests:
        facts = [make_fact(rng, next_fact_idx + i, args.preset) for i in range(args.request_size)]
        fact_by_id = {fact.fact_id: fact for fact in facts}
        for attempt in range(args.max_api_attempts):
            try:
                items = call_responses_api(
                    api_key=api_key,
                    model=args.model,
                    facts=facts,
                    preset=args.preset,
                    max_output_tokens=args.max_output_tokens,
                    reasoning_effort=args.reasoning_effort,
                    timeout=args.timeout,
                )
                break
            except urllib.error.HTTPError as exc:
                message = parse_http_error(exc)
                if exc.code == 429:
                    retry_after = parse_retry_after(exc.headers)
                    sleep_s = retry_after if retry_after is not None else args.rate_limit_sleep * (2 ** min(attempt, 3))
                    print(
                        f"Rate limited with HTTP 429 on attempt {attempt + 1}/{args.max_api_attempts}; "
                        f"sleeping {sleep_s:.1f}s ({rate_limit_summary(exc.headers)}). Message: {message}",
                        flush=True,
                    )
                    if attempt == args.max_api_attempts - 1:
                        raise
                    time.sleep(sleep_s)
                    continue
                if attempt == args.max_api_attempts - 1:
                    raise RuntimeError(f"OpenAI API HTTP {exc.code}: {message}") from exc
                sleep_s = 2.0 * (attempt + 1)
                print(
                    f"HTTP {exc.code} on attempt {attempt + 1}/{args.max_api_attempts}; "
                    f"retrying after {sleep_s:.1f}s. Message: {message}",
                    flush=True,
                )
                time.sleep(sleep_s)
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                if attempt == args.max_api_attempts - 1:
                    raise
                print(f"Request failed on attempt {attempt + 1}; retrying: {type(exc).__name__}", flush=True)
                time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError("unreachable")

        accepted = 0
        for item in items:
            try:
                text = validate_generated_item(item, fact_by_id)
            except ValueError as exc:
                print(f"Skipping invalid API item: {exc}", flush=True)
                continue
            fact = fact_by_id[item["fact_id"]]
            tok_count = token_count(tokenizer, text)
            rows.append(
                build_row(
                    example_id=next_example_id,
                    text=text,
                    fact=fact,
                    model=args.model,
                    preset=args.preset,
                    tok_count=tok_count,
                    seed=args.seed,
                )
            )
            next_example_id += 1
            total_tokens += tok_count
            accepted += 1
            if total_tokens >= args.target_tokens:
                break
        next_fact_idx += args.request_size
        requests_made += 1
        write_rows(rows, args.out, args.jsonl_out)
        print(
            f"request={requests_made} accepted={accepted} rows={len(rows)} "
            f"tokens={total_tokens}/{args.target_tokens}",
            flush=True,
        )
        time.sleep(args.sleep)

    write_rows(rows, args.out, args.jsonl_out)
    print(f"Wrote {len(rows)} rows and {total_tokens} tokens to {args.out}", flush=True)


if __name__ == "__main__":
    main()
