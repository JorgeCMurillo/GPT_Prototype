#!/usr/bin/env python3
"""Synthetic EWoK-style spatial eval for the spatial-synth experiments."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from research.bos_aligned_proto.spatial_synth.generate_spatial_relations_csv import (
    AGENTS,
    HORIZONTAL,
    OBJECTS,
    WORLD_DIRS,
    State,
    compute_relations,
    get_mention,
    neg,
    rel_phrase,
    scale,
    turn_facing,
    world_phrase,
    world_relation,
)

TIERS = ("in_format", "paraphrase", "composition")
CONCEPTS = ("left/right", "front/behind", "north/south", "east/west", "above/below", "close/far")

CONCEPT_RELATIONS: Dict[str, Tuple[str, str]] = {
    "left/right": ("left", "right"),
    "front/behind": ("front", "behind"),
    "north/south": ("north", "south"),
    "east/west": ("east", "west"),
    "above/below": ("above", "below"),
    "close/far": ("close", "far"),
}

INVERSE_RELATION: Dict[str, str] = {
    "left": "right",
    "right": "left",
    "front": "behind",
    "behind": "front",
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
    "above": "below",
    "below": "above",
    "close": "far",
    "far": "close",
}

CARDINAL_CLUES = {
    "north": "toward the map's north edge",
    "south": "toward the map's south edge",
    "east": "toward the sunrise side of the map",
    "west": "toward the sunset side of the map",
}

VERTICAL_CLUES = {
    "above": "on the higher shelf",
    "below": "on the lower shelf",
}


@dataclass(frozen=True)
class SyntheticSpatialEvalItem:
    item_id: int
    tier: str
    concept: str
    template_family: str
    contrast_type: str
    context1: str
    target1: str
    context2: str
    target2: str
    relation1: str
    relation2: str
    metadata: Dict[str, object]

    def to_row(self) -> Dict[str, object]:
        row = {
            "item_id": self.item_id,
            "Domain": "spatial-relations",
            "tier": self.tier,
            "concept": self.concept,
            "template_family": self.template_family,
            "contrast_type": self.contrast_type,
            "Context1": self.context1,
            "Target1": self.target1,
            "Context2": self.context2,
            "Target2": self.target2,
            "relation1": self.relation1,
            "relation2": self.relation2,
        }
        row.update({f"meta_{key}": value for key, value in self.metadata.items()})
        return row


def _clean(text: str) -> str:
    return " ".join(text.strip().split())


def _cap_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _state_payload(state: State) -> Dict[str, object]:
    return asdict(state)


def _choose_relation(rng: random.Random, concept: str) -> Tuple[str, str]:
    first = rng.choice(CONCEPT_RELATIONS[concept])
    return first, INVERSE_RELATION[first]


def _relative_world_dir(facing: str, rel: str) -> str:
    if rel == "front":
        return facing
    if rel == "behind":
        return turn_facing(facing, "around")
    if rel == "left":
        return turn_facing(facing, "left")
    if rel == "right":
        return turn_facing(facing, "right")
    raise ValueError(f"Unsupported egocentric relation: {rel}")


def _target_for_relation(rel: str, reference: str) -> str:
    if rel in {"left", "right", "front", "behind"}:
        return f"{rel_phrase(rel, reference)}."
    if rel in {"north", "south", "east", "west", "above", "below"}:
        return f"{world_phrase(rel, reference)}."
    if rel == "close":
        return f"close to {reference}."
    if rel == "far":
        return f"far from {reference}."
    raise ValueError(rel)


def _paraphrase_target(rel: str, reference: str) -> str:
    if rel == "left":
        return f"on {reference}'s left side."
    if rel == "right":
        return f"on {reference}'s right side."
    if rel == "front":
        return f"ahead of {reference}."
    if rel == "behind":
        return f"behind {reference}."
    if rel in {"north", "south", "east", "west"}:
        return f"{rel} of {reference}."
    if rel == "above":
        return f"higher than {reference}."
    if rel == "below":
        return f"lower than {reference}."
    if rel == "close":
        return f"near {reference}."
    if rel == "far":
        return f"far from {reference}."
    raise ValueError(rel)


def _in_format_context(
    *,
    agent: str,
    obj: str,
    concept: str,
    rel: str,
    facing: str,
) -> Tuple[str, Dict[str, object]]:
    obj_mention = get_mention(obj, False)
    obj_sentence = _cap_first(obj_mention)
    if concept in {"left/right", "front/behind"}:
        world_dir = _relative_world_dir(facing, rel)
        state = State((0, 0, 0), WORLD_DIRS[world_dir], facing, facing)
        context = (
            f"{agent} faced {facing}. {obj_sentence} was one step "
            f"{world_phrase(world_relation(WORLD_DIRS[world_dir]), agent)}. "
            f"From {agent}'s point of view, {obj_mention} was"
        )
        return context, {"state": _state_payload(state), "world_dir": world_dir}
    if concept in {"north/south", "east/west"}:
        state = State((0, 0, 0), WORLD_DIRS[rel], facing, facing)
        context = f"{agent} stayed at the marker. {obj_sentence} was one step {CARDINAL_CLUES[rel]}. {obj_sentence} was"
        return context, {"state": _state_payload(state), "world_dir": rel}
    if concept == "above/below":
        state = State((0, 0, 0), WORLD_DIRS[rel], facing, facing)
        context = f"{agent} stayed at the marker. {obj_sentence} was {VERTICAL_CLUES[rel]}. {obj_sentence} was"
        return context, {"state": _state_payload(state), "world_dir": rel}
    if concept == "close/far":
        distance = 1 if rel == "close" else 4
        state = State((0, 0, 0), scale(WORLD_DIRS[facing], distance), facing, facing)
        gap = "one step away" if rel == "close" else "four steps away"
        context = f"{agent} checked the distance to {obj_mention}. {obj_sentence} was {gap}. {obj_sentence} was"
        return context, {"state": _state_payload(state), "distance": distance}
    raise ValueError(concept)


def _paraphrase_context(
    *,
    agent: str,
    obj: str,
    concept: str,
    rel: str,
    facing: str,
) -> Tuple[str, Dict[str, object]]:
    obj_mention = get_mention(obj, False)
    obj_sentence = _cap_first(obj_mention)
    if concept in {"left/right", "front/behind"}:
        world_dir = _relative_world_dir(facing, rel)
        state = State((0, 0, 0), WORLD_DIRS[world_dir], facing, facing)
        context = (
            f"{agent} looked {facing}. {obj_sentence} occupied the "
            f"{world_relation(WORLD_DIRS[world_dir])} square from {agent}. "
            f"Relative to that gaze, it ended up"
        )
        return context, {"state": _state_payload(state), "world_dir": world_dir}
    if concept in {"north/south", "east/west"}:
        state = State((0, 0, 0), WORLD_DIRS[rel], facing, facing)
        context = (
            f"On the map, {agent} was the reference point. {obj_sentence} occupied the adjacent tile "
            f"{CARDINAL_CLUES[rel]}. Its location was"
        )
        return context, {"state": _state_payload(state), "world_dir": rel}
    if concept == "above/below":
        state = State((0, 0, 0), WORLD_DIRS[rel], facing, facing)
        context = f"In the height sketch, {obj_mention} ended {VERTICAL_CLUES[rel]}. Its level was"
        return context, {"state": _state_payload(state), "world_dir": rel}
    if concept == "close/far":
        distance = 1 if rel == "close" else 4
        state = State((0, 0, 0), scale(WORLD_DIRS[facing], distance), facing, facing)
        gap = "a single step" if rel == "close" else "four long steps"
        context = f"Judging only the gap from {agent} to {obj_mention}, the distance measured {gap}. {obj_sentence} was"
        return context, {"state": _state_payload(state), "distance": distance}
    raise ValueError(concept)


def _turn_context_for_after_relation(
    *,
    agent: str,
    obj: str,
    desired_rel: str,
    rng: random.Random,
) -> Tuple[str, Dict[str, object]]:
    candidates = []
    for facing in HORIZONTAL:
        for start_rel in ("front", "behind"):
            initial_delta = WORLD_DIRS[facing] if start_rel == "front" else neg(WORLD_DIRS[facing])
            before = State((0, 0, 0), initial_delta, facing, facing)
            for turn in ("left", "right"):
                after = State(before.a_pos, before.b_pos, turn_facing(facing, turn), before.b_facing)
                if compute_relations(after, "a")[1] == desired_rel:
                    candidates.append((before, after, start_rel, turn, facing))
    before, after, start_rel, turn, facing = rng.choice(candidates)
    obj_mention = get_mention(obj, False)
    obj_sentence = _cap_first(obj_mention)
    context = (
        f"{obj_sentence} began {rel_phrase(start_rel, agent)}. {agent} pivoted {turn} in place "
        f"while {obj_mention} stayed fixed. After the pivot, {obj_mention} was"
    )
    return context, {
        "before_state": _state_payload(before),
        "after_state": _state_payload(after),
        "start_relation": start_rel,
        "turn": turn,
        "facing": facing,
    }


def _composition_context(
    *,
    agent: str,
    obj: str,
    concept: str,
    rel: str,
    rng: random.Random,
) -> Tuple[str, Dict[str, object]]:
    obj_mention = get_mention(obj, False)
    if concept == "left/right":
        context, meta = _turn_context_for_after_relation(agent=agent, obj=obj, desired_rel=rel, rng=rng)
        meta["composition_case"] = "turn_front_back_to_left_right"
        return context, meta
    if concept == "front/behind":
        if rel == "behind":
            before = State((0, -2, 0), (0, 0, 0), "north", "north")
            after = State((0, 2, 0), (0, 0, 0), "north", "north")
            context = (
                f"{agent} faced north. {_cap_first(obj_mention)} started ahead on the path. "
                f"{agent} walked past it and stopped beyond it. At the end, {obj_mention} was"
            )
        else:
            before = State((0, -2, 0), (0, 0, 0), "north", "north")
            after = before
            context = (
                f"{agent} faced north. {_cap_first(obj_mention)} waited on the path ahead before "
                f"{agent} reached it. At that moment, {obj_mention} was"
            )
        return context, {
            "before_state": _state_payload(before),
            "after_state": _state_payload(after),
            "composition_case": "pass_through_front_behind",
        }
    if concept in {"north/south", "east/west"}:
        move_vec = WORLD_DIRS[rel]
        after = State(scale(move_vec, 2), (0, 0, 0), "north", "north")
        context = (
            f"{agent} started beside {obj_mention}. {agent} walked two steps {rel} "
            f"while {obj_mention} stayed fixed. At the end, {agent} was"
        )
        return context, {"after_state": _state_payload(after), "composition_case": "cardinal_motion_from_marker"}
    if concept == "above/below":
        after = State(WORLD_DIRS[rel], (0, 0, 0), "north", "north")
        move_word = "climbed up" if rel == "above" else "climbed down"
        context = (
            f"{agent} and {obj_mention} began level. {agent} {move_word} while "
            f"{obj_mention} stayed at the old height. Afterward, {agent} was"
        )
        return context, {"after_state": _state_payload(after), "composition_case": "vertical_inverse_height"}
    if concept == "close/far":
        distance = 1 if rel == "close" else 4
        state = State((0, 0, 0), scale(WORLD_DIRS["east"], distance), "north", "north")
        context = (
            f"The distance was checked both ways. {agent} stood {distance} step"
            f"{'' if distance == 1 else 's'} from {obj_mention}. From {agent} to {obj_mention}, the gap was"
        )
        return context, {"state": _state_payload(state), "distance": distance, "composition_case": "distance_reciprocal"}
    raise ValueError(concept)


def _make_eval_item(
    *,
    item_id: int,
    tier: str,
    concept: str,
    rng: random.Random,
) -> SyntheticSpatialEvalItem:
    agent = rng.choice(AGENTS)
    obj = rng.choice(OBJECTS)
    facing = rng.choice(HORIZONTAL)
    rel1, rel2 = _choose_relation(rng, concept)

    if tier == "in_format":
        context1, meta1 = _in_format_context(agent=agent, obj=obj, concept=concept, rel=rel1, facing=facing)
        context2, meta2 = _in_format_context(agent=agent, obj=obj, concept=concept, rel=rel2, facing=facing)
        target_fn = _target_for_relation
        template_family = "synthetic_eval_in_format"
        contrast_type = "heldout_training_style"
    elif tier == "paraphrase":
        context1, meta1 = _paraphrase_context(agent=agent, obj=obj, concept=concept, rel=rel1, facing=facing)
        context2, meta2 = _paraphrase_context(agent=agent, obj=obj, concept=concept, rel=rel2, facing=facing)
        target_fn = _paraphrase_target
        template_family = "synthetic_eval_paraphrase"
        contrast_type = "heldout_paraphrase"
    elif tier == "composition":
        context1, meta1 = _composition_context(agent=agent, obj=obj, concept=concept, rel=rel1, rng=rng)
        context2, meta2 = _composition_context(agent=agent, obj=obj, concept=concept, rel=rel2, rng=rng)
        target_fn = _target_for_relation
        template_family = f"synthetic_eval_composition_{meta1.get('composition_case', 'mixed')}"
        contrast_type = str(meta1.get("composition_case", "composition"))
    else:
        raise ValueError(f"Unsupported tier: {tier}")

    reference = agent if concept in {"left/right", "front/behind", "above/below", "close/far"} else agent
    if tier == "composition" and concept in {"north/south", "east/west", "above/below", "close/far"}:
        reference = get_mention(obj, False)
    target1 = target_fn(rel1, reference)
    target2 = target_fn(rel2, reference)
    if target1 == target2:
        raise RuntimeError(f"Synthetic eval produced identical targets for {concept}: {target1!r}")

    return SyntheticSpatialEvalItem(
        item_id=item_id,
        tier=tier,
        concept=concept,
        template_family=template_family,
        contrast_type=contrast_type,
        context1=_clean(context1),
        target1=_clean(target1),
        context2=_clean(context2),
        target2=_clean(target2),
        relation1=rel1,
        relation2=rel2,
        metadata={
            "agent": agent,
            "object": obj,
            "facing": facing,
            "context1_meta": meta1,
            "context2_meta": meta2,
        },
    )


def _balanced_concept_sequence(n: int) -> List[str]:
    if n <= 0:
        return []
    base, remainder = divmod(n, len(CONCEPTS))
    out: List[str] = []
    for idx, concept in enumerate(CONCEPTS):
        out.extend([concept] * (base + (1 if idx < remainder else 0)))
    return out


def generate_synthetic_spatial_eval_items(
    *,
    n_per_tier: int = 300,
    seed: int = 100000,
    template_preset: str = "v4",
) -> List[SyntheticSpatialEvalItem]:
    """Generate a balanced held-out synthetic spatial completion-choice eval.

    ``template_preset`` is recorded for provenance. The eval intentionally uses
    its own wording so it can test transfer away from the training CSV without
    copying EWoK templates.
    """
    if n_per_tier <= 0:
        return []
    rng = random.Random(seed)
    items: List[SyntheticSpatialEvalItem] = []
    for tier in TIERS:
        concepts = _balanced_concept_sequence(n_per_tier)
        rng.shuffle(concepts)
        for concept in concepts:
            item = _make_eval_item(item_id=len(items), tier=tier, concept=concept, rng=rng)
            meta = dict(item.metadata)
            meta["template_preset"] = template_preset
            items.append(
                SyntheticSpatialEvalItem(
                    **{**asdict(item), "metadata": meta},
                )
            )
    return items


def write_synthetic_spatial_eval_dataset(items: Iterable[SyntheticSpatialEvalItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_row(), ensure_ascii=False) + "\n")


def _reduce_token_logps(token_logps: torch.Tensor, score_reduction: str) -> float:
    if token_logps.numel() == 0:
        return 0.0
    if score_reduction == "sum":
        return float(token_logps.sum().item())
    if score_reduction == "mean":
        return float(token_logps.mean().item())
    raise ValueError(f"score_reduction must be 'sum' or 'mean', got {score_reduction!r}")


def _score_pairs(
    model,
    tokenizer,
    contexts: Sequence[str],
    targets: Sequence[str],
    *,
    batch_size: int,
    score_reduction: str,
):
    from evaluation.ewok import per_token_conditional_log_likelihood

    token_logps = per_token_conditional_log_likelihood(
        model,
        tokenizer,
        list(contexts),
        list(targets),
        batch_size=batch_size,
    )
    return [_reduce_token_logps(row, score_reduction) for row in token_logps]


def summarize_synthetic_spatial_records(records: Sequence[Dict[str, object]], *, margin_eps: float = 1e-6) -> Dict:
    def stats(subset: Sequence[Dict[str, object]]) -> Dict[str, float | int]:
        if not subset:
            return {
                "n": 0,
                "acc_official": 0.0,
                "acc_symmetric": 0.0,
                "acc_combined": 0.0,
                "mean_signed_m": 0.0,
                "mean_abs_m": 0.0,
                "tie_rate_m": 0.0,
            }
        margins = [float(row["margin_combined"]) for row in subset]
        return {
            "n": len(subset),
            "acc_official": sum(1.0 if row["correct_official"] else 0.0 for row in subset) / len(subset),
            "acc_symmetric": sum(1.0 if row["correct_symmetric"] else 0.0 for row in subset) / len(subset),
            "acc_combined": sum(float(row["correct_combined"]) for row in subset) / len(subset),
            "mean_signed_m": sum(margins) / len(margins),
            "mean_abs_m": sum(abs(x) for x in margins) / len(margins),
            "tie_rate_m": sum(1.0 if abs(x) < margin_eps else 0.0 for x in margins) / len(margins),
        }

    def by_key(key: str) -> Dict[str, Dict[str, float | int]]:
        buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for record in records:
            buckets[str(record[key])].append(record)
        return {name: stats(rows) for name, rows in sorted(buckets.items())}

    return {
        "overall": stats(records),
        "by_tier": by_key("tier"),
        "by_concept": by_key("concept"),
        "by_template_family": by_key("template_family"),
        "by_contrast_type": by_key("contrast_type"),
    }


def evaluate_synthetic_spatial(
    *,
    model,
    tokenizer,
    items: Sequence[SyntheticSpatialEvalItem],
    batch_size: int = 8,
    score_reduction: str = "mean",
    margin_eps: float = 1e-6,
) -> Tuple[Dict, List[Dict[str, object]]]:
    if not items:
        summary = summarize_synthetic_spatial_records([], margin_eps=margin_eps)
        return summary, []

    contexts1 = [item.context1 for item in items]
    targets1 = [item.target1 for item in items]
    contexts2 = [item.context2 for item in items]
    targets2 = [item.target2 for item in items]

    with torch.no_grad():
        s11 = _score_pairs(
            model,
            tokenizer,
            contexts1,
            targets1,
            batch_size=batch_size,
            score_reduction=score_reduction,
        )
        s12 = _score_pairs(
            model,
            tokenizer,
            contexts1,
            targets2,
            batch_size=batch_size,
            score_reduction=score_reduction,
        )
        s22 = _score_pairs(
            model,
            tokenizer,
            contexts2,
            targets2,
            batch_size=batch_size,
            score_reduction=score_reduction,
        )
        s21 = _score_pairs(
            model,
            tokenizer,
            contexts2,
            targets1,
            batch_size=batch_size,
            score_reduction=score_reduction,
        )

    per_item: List[Dict[str, object]] = []
    for idx, item in enumerate(items):
        m1 = float(s11[idx] - s12[idx])
        m2 = float(s22[idx] - s21[idx])
        m = 0.5 * (m1 + m2)
        correct1 = m1 > 0.0
        correct2 = m2 > 0.0
        rec = item.to_row()
        rec.update(
            {
                "score_reduction": score_reduction,
                "S11_logp_T1_given_C1": float(s11[idx]),
                "S12_logp_T2_given_C1": float(s12[idx]),
                "S22_logp_T2_given_C2": float(s22[idx]),
                "S21_logp_T1_given_C2": float(s21[idx]),
                "margin_official_m1": m1,
                "margin_symmetric_m2": m2,
                "margin_combined": m,
                "correct_official": bool(correct1),
                "correct_symmetric": bool(correct2),
                "correct_combined": 0.5 * (float(correct1) + float(correct2)),
                "near_tie_official": bool(abs(m1) < margin_eps),
                "near_tie_symmetric": bool(abs(m2) < margin_eps),
                "near_tie_combined": bool(abs(m) < margin_eps),
            }
        )
        per_item.append(rec)

    return summarize_synthetic_spatial_records(per_item, margin_eps=margin_eps), per_item
