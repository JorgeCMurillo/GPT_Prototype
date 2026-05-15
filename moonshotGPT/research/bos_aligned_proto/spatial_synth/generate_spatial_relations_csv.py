#!/usr/bin/env python3
"""Generate synthetic spatial-relations causal-LM text as CSV.

The generator keeps the small 3D world from the exploratory snippet: an agent
and an object occupy one-step axis-aligned positions, the observer can be either
entity, and the text states world-relative plus observer-relative relations
before and after a change.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

Vec = Tuple[int, int, int]

WORLD_DIRS: Dict[str, Vec] = {
    "east": (1, 0, 0),
    "west": (-1, 0, 0),
    "north": (0, 1, 0),
    "south": (0, -1, 0),
    "above": (0, 0, 1),
    "below": (0, 0, -1),
}

INVERSE_REL: Dict[str, str] = {
    "above": "below",
    "below": "above",
    "left": "right",
    "right": "left",
    "front": "behind",
    "behind": "front",
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
    "level": "level",
}

HORIZONTAL = ["north", "east", "south", "west"]
CARDINAL = ["north", "south", "east", "west"]

AGENTS = ["Ava", "Noah", "Mira", "Kai", "Lena", "Omar", "Jay", "Mohammed", "Leo", "Ali"]
OBJECTS = ["lantern", "statue", "pig", "mug", "book", "box", "key", "stone", "cow"]

DIFFICULTY_TO_LABEL = {1: "easy", 2: "medium", 3: "hard"}
LABEL_TO_DIFFICULTY = {v: k for k, v in DIFFICULTY_TO_LABEL.items()}
DIFFICULTY_LABELS = ("easy", "medium", "hard")

# v5 note: left/right likely needs contrastive facing pairs because the rule is
# world direction + observer facing -> egocentric side, not just word exposure.
# If left/right margins keep falling, the next targeted families should separate
# three skills that EWoK probes but v5 only partly covers: (1) turn-around side
# flips, e.g. right before the turn becomes left after it; (2) left/right turns
# from front/back landmarks, e.g. an object in front becomes right after a left
# turn; and (3) subject/reference reciprocals, e.g. A left of B means B right of
# A. Keep these implicit and lexicalized differently from EWoK templates.
TEMPLATE_PRESETS: Dict[str, Tuple[List[str], List[int]]] = {
    "v3": (["explicit", "implicit_vertical", "implicit_turn", "implicit_pass_by"], [5, 2, 2, 2]),
    "v4": (
        ["explicit", "implicit_vertical", "implicit_turn", "implicit_pass_by", "implicit_pass_through"],
        [5, 2, 2, 2, 2],
    ),
    "v5": (
        [
            "explicit",
            "implicit_vertical",
            "implicit_turn",
            "implicit_pass_by",
            "implicit_pass_through",
            "implicit_left_right_contrast",
        ],
        [5, 2, 2, 2, 2, 2],
    ),
    "v6": (
        [
            "explicit",
            "implicit_vertical",
            "implicit_turn",
            "implicit_pass_by",
            "implicit_pass_through",
            "implicit_left_right_contrast",
            "implicit_turn_lr_contrast",
        ],
        [5, 2, 2, 2, 2, 2, 2],
    ),
    "v7": (
        [
            "explicit",
            "implicit_vertical",
            "implicit_turn",
            "implicit_pass_by",
            "implicit_pass_through",
            "implicit_left_right_contrast",
            "implicit_turn_lr_contrast",
            "implicit_distance_contrast",
        ],
        [5, 2, 2, 2, 2, 2, 2, 2],
    ),
    "v8": (
        [
            "explicit",
            "implicit_vertical",
            "implicit_turn",
            "implicit_pass_by",
            "implicit_pass_through",
            "implicit_left_right_contrast",
            "implicit_turn_lr_contrast",
            "implicit_distance_reciprocal",
        ],
        [5, 2, 2, 2, 2, 2, 2, 2],
    ),
    "v9": (
        [
            "explicit",
            "implicit_vertical",
            "implicit_turn",
            "implicit_pass_by",
            "implicit_pass_through",
            "implicit_left_right_contrast",
            "implicit_turn_lr_contrast",
            "implicit_distance_contrast",
        ],
        [5, 2, 2, 2, 2, 2, 2, 1],
    ),
    "v10": (
        [
            "explicit",
            "implicit_vertical",
            "implicit_turn",
            "implicit_pass_by",
            "implicit_pass_through",
            "implicit_left_right_contrast",
            "implicit_turn_lr_contrast",
            "implicit_turn_around_lr",
        ],
        [5, 2, 2, 2, 2, 2, 2, 2],
    ),
    "v11": (
        [
            "explicit",
            "implicit_vertical",
            "implicit_turn",
            "implicit_pass_by",
            "implicit_pass_through",
            "implicit_left_right_contrast",
            "implicit_turn_lr_contrast",
            "implicit_cardinal_guard",
        ],
        [5, 2, 2, 2, 2, 2, 2, 2],
    ),
    "v12": (
        [
            "explicit",
            "implicit_vertical",
            "implicit_turn",
            "implicit_pass_by",
            "implicit_pass_through",
            "implicit_left_right_contrast",
            "implicit_turn_lr_contrast",
            "implicit_relation_type_contrast",
        ],
        [5, 2, 2, 2, 2, 2, 2, 2],
    ),
    "v13": (
        [
            "explicit",
            "implicit_vertical",
            "implicit_turn",
            "implicit_pass_by",
            "implicit_pass_through",
            "implicit_left_right_contrast",
            "implicit_turn_lr_contrast",
            "implicit_distance_reciprocal",
            "implicit_turn_lr_order_variants",
        ],
        [5, 2, 2, 2, 2, 2, 2, 2, 2],
    ),
    "v14": (
        [
            "explicit",
            "implicit_vertical",
            "implicit_turn",
            "implicit_pass_by",
            "implicit_pass_through",
            "implicit_left_right_contrast",
            "implicit_turn_lr_contrast",
            "implicit_distance_reciprocal",
            "implicit_turn_around_lr",
        ],
        [5, 2, 2, 2, 2, 2, 2, 2, 2],
    ),
    "v15": (
        [
            "explicit",
            "implicit_vertical",
            "implicit_turn",
            "implicit_pass_by",
            "implicit_pass_through",
            "implicit_left_right_contrast",
            "implicit_turn_lr_contrast",
            "implicit_distance_reciprocal",
            "implicit_turn_around_lr",
            "implicit_turn_lr_order_variants",
        ],
        [5, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ),
    "v19": (
        [
            "explicit",
            "implicit_vertical",
            "implicit_turn",
            "implicit_pass_by",
            "implicit_pass_through",
            "implicit_left_right_contrast",
            "implicit_turn_lr_contrast",
            "implicit_distance_reciprocal",
            "implicit_turn_around_lr",
            "implicit_left_right_paired_contrast",
        ],
        [5, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ),
    "v20": (
        [
            "explicit",
            "implicit_vertical",
            "implicit_turn",
            "implicit_pass_by",
            "implicit_pass_through",
            "implicit_left_right_contrast",
            "implicit_turn_lr_contrast",
            "implicit_distance_reciprocal",
            "implicit_turn_around_lr",
            "implicit_left_right_paired_contrast",
        ],
        [5, 2, 2, 2, 2, 2, 2, 2, 2, 4],
    ),
    "v21": (
        [
            "explicit",
            "implicit_vertical",
            "implicit_turn",
            "implicit_pass_by",
            "implicit_pass_through",
            "implicit_left_right_contrast",
            "implicit_turn_lr_contrast",
            "implicit_distance_reciprocal",
            "implicit_turn_around_lr",
            "implicit_left_right_paired_contrast",
        ],
        [5, 2, 2, 2, 2, 2, 2, 2, 2, 6],
    ),
    # Cardinal-only preset for natural-data mixing tests. The goal is to target
    # north/south and east/west without adding more left/right, vertical, or
    # distance examples, so we can test whether small token-budget injections
    # improve those EWoK slices without broad spatial distribution narrowing.
    "cardinal_v1": (
        [
            "implicit_cardinal_inverse",
            "implicit_cardinal_move_past",
            "implicit_cardinal_landmark",
            "implicit_cardinal_route_update",
        ],
        [2, 2, 2, 2],
    ),
}

TEMPLATES = [
    (
        "{T} started one step {before_world}. {O} faced {before_facing}, so from "
        "{O_poss} point of view, {T} was {before_rel}. {op} Afterward, {T} was "
        "{after_world}. From {O_poss} point of view, {T} was {after_rel}."
    ),
    (
        "From {O_poss} point of view, {T} began {before_rel}. {T} was "
        "{before_world}, and {O} faced {before_facing}. {op} Later, from "
        "{O_poss} point of view, {T} was {after_rel}. {T} was {after_world}."
    ),
    (
        "Before the change, {O} faced {before_facing}. {T} was {before_world}, "
        "which put {T} {before_rel} from {O_poss} point of view. {op} After "
        "the change, {T} was {after_world}, which put {T} {after_rel} from "
        "{O_poss} point of view."
    ),
    (
        "{O} faced {before_facing}. At first, {T} was {before_world} and "
        "{before_rel} from {O_poss} point of view. {op} Afterward, {T} was "
        "{after_world} and {after_rel} from {O_poss} point of view."
    ),
    (
        "{O} faced {before_facing}. {T} was {before_world}. From {O_poss} view, "
        "{T} was {before_rel}. {op} Then {T} was {after_world}. From {O_poss} "
        "view, {T} was {after_rel}."
    ),
]


@dataclass(frozen=True)
class State:
    a_pos: Vec
    b_pos: Vec
    a_facing: str
    b_facing: str


@dataclass(frozen=True)
class Item:
    example_id: int
    agent: str
    obj: str
    operation: str
    observer: str
    before_state: State
    after_state: State
    before_world: str
    before_relative: str
    after_world: str
    after_relative: str
    text: str
    difficulty: int
    difficulty_label: str
    template_id: int
    template_family: str


def add(u: Vec, v: Vec) -> Vec:
    return (u[0] + v[0], u[1] + v[1], u[2] + v[2])


def sub(u: Vec, v: Vec) -> Vec:
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])


def neg(u: Vec) -> Vec:
    return (-u[0], -u[1], -u[2])


def scale(u: Vec, k: int) -> Vec:
    return (u[0] * k, u[1] * k, u[2] * k)


def turn_facing(facing: str, turn: str) -> str:
    i = HORIZONTAL.index(facing)
    if turn == "left":
        return HORIZONTAL[(i - 1) % 4]
    if turn == "right":
        return HORIZONTAL[(i + 1) % 4]
    if turn == "around":
        return HORIZONTAL[(i + 2) % 4]
    raise ValueError(f"Unknown turn: {turn}")


def world_relation(delta: Vec) -> str:
    if delta == (0, 0, 0):
        return "level"
    x, y, z = delta
    if y == 0 and z == 0 and x != 0:
        return "east" if x > 0 else "west"
    if x == 0 and z == 0 and y != 0:
        return "north" if y > 0 else "south"
    if x == 0 and y == 0 and z != 0:
        return "above" if z > 0 else "below"
    raise ValueError(f"Delta is not axis aligned: {delta}")


def relative_relation(delta: Vec, facing: str) -> str:
    """Return the target relation from the observer's facing direction."""
    world_rel = world_relation(delta)
    if world_rel == "level":
        return "level"
    if world_rel == "above":
        return "above"
    if world_rel == "below":
        return "below"

    unit_delta = WORLD_DIRS[world_rel]
    front = WORLD_DIRS[facing]
    back = neg(front)
    left = WORLD_DIRS[turn_facing(facing, "left")]
    right = WORLD_DIRS[turn_facing(facing, "right")]

    if unit_delta == front:
        return "front"
    if unit_delta == back:
        return "behind"
    if unit_delta == left:
        return "left"
    if unit_delta == right:
        return "right"

    raise ValueError(f"Cannot compute relative relation for delta={delta}, facing={facing}")


def rel_phrase(rel: str, target: str) -> str:
    if rel == "left":
        return f"to {target}'s left"
    if rel == "right":
        return f"to {target}'s right"
    if rel == "front":
        return f"in front of {target}"
    if rel == "behind":
        return f"behind {target}"
    if rel == "above":
        return f"above {target}"
    if rel == "below":
        return f"below {target}"
    if rel == "level":
        return f"level with {target}"
    raise ValueError(rel)


def egocentric_look_action(rel: str) -> str:
    if rel == "front":
        return "look straight ahead"
    if rel == "behind":
        return "look back over a shoulder"
    if rel == "left":
        return "look to the left"
    if rel == "right":
        return "look to the right"
    raise ValueError(rel)


def egocentric_view_region(rel: str, observer_possessive: str) -> str:
    if rel == "front":
        return f"the forward part of {observer_possessive} view"
    if rel == "behind":
        return f"the space behind {observer_possessive} stance"
    if rel == "left":
        return f"the left-hand side of {observer_possessive} view"
    if rel == "right":
        return f"the right-hand side of {observer_possessive} view"
    raise ValueError(rel)


def world_phrase(rel: str, target: str) -> str:
    if rel in {"north", "south", "east", "west"}:
        return f"{rel} of {target}"
    if rel in {"above", "below"}:
        return f"{rel} {target}"
    if rel == "level":
        return f"level with {target}"
    raise ValueError(rel)


def inverse_relation(rel: str) -> str:
    try:
        return INVERSE_REL[rel]
    except KeyError as exc:
        raise ValueError(f"No inverse relation for {rel!r}") from exc


def compute_relations(state: State, observer: str) -> Tuple[str, str]:
    if observer == "a":
        delta = sub(state.b_pos, state.a_pos)
        return world_relation(delta), relative_relation(delta, state.a_facing)
    if observer == "b":
        delta = sub(state.a_pos, state.b_pos)
        return world_relation(delta), relative_relation(delta, state.b_facing)
    raise ValueError(f"Unknown observer: {observer}")


def apply_operation(state: State, rng: random.Random) -> Tuple[State, str]:
    kind = rng.choice(["rotate_agent", "move_object", "move_both", "noop"])

    if kind == "rotate_agent":
        turn = rng.choice(["left", "right", "around"])
        new_facing = turn_facing(state.a_facing, turn)
        return State(state.a_pos, state.b_pos, new_facing, state.b_facing), f"agent_turns_{turn}"

    if kind == "move_object":
        current_delta = sub(state.b_pos, state.a_pos)
        possible = [v for v in WORLD_DIRS.values() if v != current_delta]
        new_delta = rng.choice(possible)
        new_state = State(state.a_pos, add(state.a_pos, new_delta), state.a_facing, state.b_facing)
        return new_state, f"object_moves_to_{world_relation(new_delta)}_side"

    if kind == "move_both":
        move_dir = rng.choice(["north", "south", "east", "west"])
        step = WORLD_DIRS[move_dir]
        new_state = State(add(state.a_pos, step), add(state.b_pos, step), state.a_facing, state.b_facing)
        return new_state, f"both_move_{move_dir}"

    return State(state.a_pos, state.b_pos, state.a_facing, state.b_facing), "nothing_changes"


def get_mention(name: str, is_agent: bool) -> str:
    return name if is_agent else f"the {name}"


def get_possessive(name: str, is_agent: bool) -> str:
    return f"{name}'s" if is_agent else f"the {name}'s"


def get_operation_text(agent: str, obj: str, operation: str, after_state: State) -> str:
    if operation.startswith("agent_turns_"):
        turn = operation.replace("agent_turns_", "")
        return f"Then {agent} turned {turn}."
    if operation.startswith("object_moves_to_"):
        new_delta = sub(after_state.b_pos, after_state.a_pos)
        side = world_relation(new_delta)
        return f"Then the {obj} changed position and ended up one step {world_phrase(side, agent)}."
    if operation.startswith("both_move_"):
        direction = operation.replace("both_move_", "")
        return f"Then {agent} and the {obj} both moved one step {direction}."
    if operation == "nothing_changes":
        return "Then nothing moved or turned."
    raise ValueError(operation)


def _remove_vertical_relative_repetition(
    text: str,
    *,
    world_rel: str,
    relative_phrase: str,
    observer_possessive: str,
    target_mention: str,
    before: bool,
) -> str:
    if world_rel not in {"above", "below"}:
        return text

    replacements = [
        f" and {relative_phrase} from {observer_possessive} point of view",
        f", which put {target_mention} {relative_phrase} from {observer_possessive} point of view",
        f". From {observer_possessive} view, {target_mention} was {relative_phrase}",
    ]
    if before:
        replacements.extend(
            [
                f", so from {observer_possessive} point of view, {target_mention} was {relative_phrase}",
                f"From {observer_possessive} point of view, {target_mention} began {relative_phrase}. ",
            ]
        )
    else:
        replacements.extend(
            [
                f". From {observer_possessive} point of view, {target_mention} was {relative_phrase}",
                f" Later, from {observer_possessive} point of view, {target_mention} was {relative_phrase}. ",
            ]
        )

    for needle in replacements:
        text = text.replace(needle, " Later, " if (not before and needle.startswith(" Later")) else "")
    return text


def render_text(
    agent: str,
    obj: str,
    before: State,
    after: State,
    operation: str,
    observer: str,
    rng: random.Random,
) -> Tuple[str, int, int]:
    before_world, before_rel = compute_relations(before, observer)
    after_world, after_rel = compute_relations(after, observer)

    is_agent_obs = observer == "a"
    obs_name = agent if is_agent_obs else obj
    target_name = obj if is_agent_obs else agent

    obs_mention = get_mention(obs_name, is_agent_obs)
    target_mention = get_mention(target_name, not is_agent_obs)
    obs_possessive = get_possessive(obs_name, is_agent_obs)
    obs_facing = before.a_facing if is_agent_obs else before.b_facing

    template_id = rng.randrange(len(TEMPLATES))
    text = TEMPLATES[template_id].format(
        O=obs_mention,
        T=target_mention,
        O_poss=obs_possessive,
        before_facing=obs_facing,
        before_world=world_phrase(before_world, obs_mention),
        before_rel=rel_phrase(before_rel, obs_mention),
        after_world=world_phrase(after_world, obs_mention),
        after_rel=rel_phrase(after_rel, obs_mention),
        op=get_operation_text(agent, obj, operation, after),
    )

    text = _remove_vertical_relative_repetition(
        text,
        world_rel=before_world,
        relative_phrase=rel_phrase(before_rel, obs_mention),
        observer_possessive=obs_possessive,
        target_mention=target_mention,
        before=True,
    )
    text = _remove_vertical_relative_repetition(
        text,
        world_rel=after_world,
        relative_phrase=rel_phrase(after_rel, obs_mention),
        observer_possessive=obs_possessive,
        target_mention=target_mention,
        before=False,
    )

    sentences = text.split(". ")
    text = ". ".join(s[0].upper() + s[1:] if s else s for s in sentences)
    text = " ".join(text.split())

    difficulty = 2
    if operation == "nothing_changes":
        difficulty = 1
    if template_id in {2, 3}:
        difficulty = 3

    return text, difficulty, template_id


def _clean_text(text: str) -> str:
    sentences = text.split(". ")
    text = ". ".join(s[0].upper() + s[1:] if s else s for s in sentences)
    return " ".join(text.split())


def _build_item(
    *,
    example_id: int,
    agent: str,
    obj: str,
    operation: str,
    observer: str,
    before: State,
    after: State,
    text: str,
    difficulty: int,
    template_id: int,
    template_family: str,
) -> Item:
    before_world, before_rel = compute_relations(before, observer)
    after_world, after_rel = compute_relations(after, observer)
    item = Item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=operation,
        observer=observer,
        before_state=before,
        after_state=after,
        before_world=before_world,
        before_relative=before_rel,
        after_world=after_world,
        after_relative=after_rel,
        text=_clean_text(text),
        difficulty=difficulty,
        difficulty_label=DIFFICULTY_TO_LABEL[difficulty],
        template_id=template_id,
        template_family=template_family,
    )
    validate_item(item)
    return item


def generate_vertical_implicit_item(
    rng: random.Random,
    *,
    agent: str,
    obj: str,
    example_id: int = 0,
    mover: str | None = None,
    direction: str | None = None,
    relation_view: str | None = None,
) -> Item:
    """Generate vertical movement with direct or inverse final relation wording."""
    mover = mover or rng.choice(["a", "b"])
    direction = direction or rng.choice(["up", "down"])
    relation_view = relation_view or rng.choice(["direct", "inverse"])
    if mover not in {"a", "b"}:
        raise ValueError(f"mover must be 'a' or 'b', got {mover!r}")
    if direction not in {"up", "down"}:
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    if relation_view not in {"direct", "inverse"}:
        raise ValueError(f"relation_view must be 'direct' or 'inverse', got {relation_view!r}")

    before = State((0, 0, 0), (0, 0, 0), rng.choice(HORIZONTAL), rng.choice(HORIZONTAL))
    step = WORLD_DIRS["above"] if direction == "up" else WORLD_DIRS["below"]
    if mover == "a":
        after = State(step, before.b_pos, before.a_facing, before.b_facing)
        mover_name, stayer_name = agent, obj
        mover_is_agent, stayer_is_agent = True, False
    else:
        after = State(before.a_pos, step, before.a_facing, before.b_facing)
        mover_name, stayer_name = obj, agent
        mover_is_agent, stayer_is_agent = False, True

    direct_rel = "above" if direction == "up" else "below"
    if relation_view == "direct":
        subject_name, subject_is_agent = mover_name, mover_is_agent
        reference_name, reference_is_agent = stayer_name, stayer_is_agent
        final_rel = direct_rel
        observer = "b" if mover == "a" else "a"
    else:
        subject_name, subject_is_agent = stayer_name, stayer_is_agent
        reference_name, reference_is_agent = mover_name, mover_is_agent
        final_rel = inverse_relation(direct_rel)
        observer = mover

    mover_mention = get_mention(mover_name, mover_is_agent)
    stayer_mention = get_mention(stayer_name, stayer_is_agent)
    subject = get_mention(subject_name, subject_is_agent)
    reference = get_mention(reference_name, reference_is_agent)
    move_word = "higher" if direction == "up" else "lower"
    template_id = 100 + rng.randrange(3)
    templates = [
        (
            "{A} and {B} started level with each other. {mover} shifted {move_word} "
            "while {stayer} stayed in place. After the change, {subject} was {relation}."
        ),
        (
            "{A} and {B} began at matching height. Only {mover} changed height; "
            "{stayer} did not move. By the end, {subject} was {relation}."
        ),
        (
            "{A} and {B} were even at first. {mover} moved {move_word}, and "
            "{stayer} remained in place. The new arrangement left {subject} {relation}."
        ),
    ]
    text = templates[template_id - 100].format(
        A=get_mention(agent, True),
        B=get_mention(obj, False),
        mover=mover_mention,
        stayer=stayer_mention,
        move_word=move_word,
        subject=subject,
        relation=world_phrase(final_rel, reference),
    )
    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=f"implicit_vertical_{'agent' if mover == 'a' else 'object'}_moves_{direction}_{relation_view}",
        observer=observer,
        before=before,
        after=after,
        text=text,
        difficulty=3,
        template_id=template_id,
        template_family="implicit_vertical",
    )


def generate_turn_implicit_item(
    rng: random.Random,
    *,
    agent: str,
    obj: str,
    example_id: int = 0,
    turn: str | None = None,
    template_variant: int | None = None,
) -> Item:
    """Generate a reference-frame update where only the agent turns."""
    a_facing = rng.choice(HORIZONTAL)
    b_facing = rng.choice(HORIZONTAL)
    initial_delta = rng.choice([WORLD_DIRS[name] for name in HORIZONTAL])
    before = State((0, 0, 0), initial_delta, a_facing, b_facing)
    turn = turn or rng.choice(["left", "right", "around"])
    after = State(before.a_pos, before.b_pos, turn_facing(a_facing, turn), before.b_facing)
    after_rel = compute_relations(after, "a")[1]
    templates = [
        (
            "{agent} turned {turn} while {obj} stayed fixed nearby. The scene did not move, "
            "but from {agent_poss} view, {obj} ended up {after_rel}."
        ),
        (
            "{obj} stayed in the same spot as {agent} turned {turn}. Nothing slid across "
            "the floor; only {agent_poss} viewpoint changed. Afterward, {obj} was {after_rel}."
        ),
        (
            "{agent} changed direction by turning {turn}. {obj} remained where it was, "
            "so its side from {agent_poss} view changed to {after_rel}."
        ),
        (
            "{agent} paused and turned {turn} in place. {obj} kept the same spot. "
            "From the new stance, {agent} would need to {look_action} to face {obj}."
        ),
        (
            "After rotating {turn}, {agent} had a new view of the same area. {obj} "
            "had not moved; it now sat in {view_region}."
        ),
        (
            "{agent} changed stance with a {turn} turn while {obj} stayed put. The "
            "shortest glance toward {obj} was now: {look_action}."
        ),
    ]
    if template_variant is None:
        template_variant = rng.randrange(len(templates))
    if not 0 <= template_variant < len(templates):
        raise ValueError(f"template_variant must be in [0, {len(templates) - 1}], got {template_variant}")
    template_id = 200 + template_variant
    text = templates[template_variant].format(
        agent=agent,
        agent_poss=get_possessive(agent, True),
        obj=get_mention(obj, False),
        turn=turn,
        after_rel=rel_phrase(after_rel, agent),
        look_action=egocentric_look_action(after_rel),
        view_region=egocentric_view_region(after_rel, get_possessive(agent, True)),
    )
    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=f"implicit_turn_agent_{turn}",
        observer="a",
        before=before,
        after=after,
        text=text,
        difficulty=3,
        template_id=template_id,
        template_family="implicit_turn",
    )


def generate_left_right_contrast_item(
    rng: random.Random,
    *,
    agent: str,
    obj: str,
    example_id: int = 0,
    side: str | None = None,
    before_facing: str | None = None,
    template_variant: int | None = None,
) -> Item:
    """Generate a compact contrast where facing around flips left/right."""
    side = side or rng.choice(["left", "right"])
    before_facing = before_facing or rng.choice(HORIZONTAL)
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if before_facing not in HORIZONTAL:
        raise ValueError(f"before_facing must be one of {HORIZONTAL}, got {before_facing!r}")

    world_dir = turn_facing(before_facing, side)
    after_facing = turn_facing(before_facing, "around")
    before = State((0, 0, 0), WORLD_DIRS[world_dir], before_facing, rng.choice(HORIZONTAL))
    after = State(before.a_pos, before.b_pos, after_facing, before.b_facing)
    before_rel = compute_relations(before, "a")[1]
    after_rel = compute_relations(after, "a")[1]

    templates = [
        (
            "{obj} stayed one step {world_pos}. When {agent} faced {before_facing}, "
            "{obj} was {before_rel}. After {agent} faced {after_facing} without moving, "
            "the same spot was {after_rel}."
        ),
        (
            "{agent} checked the same fixed spot from two stances. With {obj} {world_pos}, "
            "facing {before_facing} put {obj} {before_rel}; facing {after_facing} put "
            "{obj} {after_rel}."
        ),
        (
            "Nothing changed places: {obj} remained {world_pos}. From a {before_facing}-facing "
            "stance it was {before_rel}, but from a {after_facing}-facing stance it was {after_rel}."
        ),
    ]
    if template_variant is None:
        template_variant = rng.randrange(len(templates))
    if not 0 <= template_variant < len(templates):
        raise ValueError(f"template_variant must be in [0, {len(templates) - 1}], got {template_variant}")
    template_id = 500 + template_variant
    obj_mention = get_mention(obj, False)
    text = templates[template_variant].format(
        agent=agent,
        obj=obj_mention,
        world_pos=world_phrase(world_relation(WORLD_DIRS[world_dir]), agent),
        before_facing=before_facing,
        after_facing=after_facing,
        before_rel=rel_phrase(before_rel, agent),
        after_rel=rel_phrase(after_rel, agent),
    )
    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=f"implicit_left_right_contrast_{before_facing}_to_{after_facing}_{side}",
        observer="a",
        before=before,
        after=after,
        text=text,
        difficulty=3,
        template_id=template_id,
        template_family="implicit_left_right_contrast",
    )


def generate_turn_lr_contrast_item(
    rng: random.Random,
    *,
    agent: str,
    obj: str,
    example_id: int = 0,
    start_relation: str | None = None,
    turn: str | None = None,
    before_facing: str | None = None,
    template_variant: int | None = None,
) -> Item:
    """Generate front/back-to-left/right examples from an in-place turn."""
    start_relation = start_relation or rng.choice(["front", "behind"])
    turn = turn or rng.choice(["left", "right"])
    before_facing = before_facing or rng.choice(HORIZONTAL)
    if start_relation not in {"front", "behind"}:
        raise ValueError(f"start_relation must be 'front' or 'behind', got {start_relation!r}")
    if turn not in {"left", "right"}:
        raise ValueError(f"turn must be 'left' or 'right', got {turn!r}")
    if before_facing not in HORIZONTAL:
        raise ValueError(f"before_facing must be one of {HORIZONTAL}, got {before_facing!r}")

    initial_delta = WORLD_DIRS[before_facing] if start_relation == "front" else neg(WORLD_DIRS[before_facing])
    before = State((0, 0, 0), initial_delta, before_facing, rng.choice(HORIZONTAL))
    after = State(before.a_pos, before.b_pos, turn_facing(before_facing, turn), before.b_facing)
    before_rel = compute_relations(before, "a")[1]
    after_rel = compute_relations(after, "a")[1]
    after_side = "left-hand" if after_rel == "left" else "right-hand"

    left_after = State(before.a_pos, before.b_pos, turn_facing(before_facing, "left"), before.b_facing)
    right_after = State(before.a_pos, before.b_pos, turn_facing(before_facing, "right"), before.b_facing)
    left_turn_rel = compute_relations(left_after, "a")[1]
    right_turn_rel = compute_relations(right_after, "a")[1]

    # Order may matter for generalization: the current v6 templates mostly say
    # turn/pivot first and then recover the original front/back placement. A
    # future variant should also test the reverse order, e.g. "in front + pivot
    # right -> left-hand side", so we can see whether models learn the spatial
    # composition instead of a preferred sentence order.
    templates = [
        (
            "{obj} stayed fixed as {agent} pivoted {turn}. Before the pivot, {obj} "
            "had been {before_rel}; after it, the same spot fell on {agent_poss} "
            "{after_side} side."
        ),
        (
            "{agent} changed which direction counted as forward with a {turn} turn. "
            "{obj} did not move from its spot {before_rel}. From the new stance, "
            "{agent} would need to {look_action} to face it."
        ),
        (
            "The same spot stayed put while {agent} turned {turn}. It started "
            "{before_rel}; once {agent} settled, it was {after_rel}."
        ),
        (
            "With {obj} held in the same place {before_rel}, {agent} compared two "
            "pivots. A left turn put that spot {left_turn_rel}; a right turn put it "
            "{right_turn_rel}."
        ),
    ]
    if template_variant is None:
        template_variant = rng.randrange(len(templates))
    if not 0 <= template_variant < len(templates):
        raise ValueError(f"template_variant must be in [0, {len(templates) - 1}], got {template_variant}")
    template_id = 600 + template_variant
    obj_mention = get_mention(obj, False)
    text = templates[template_variant].format(
        agent=agent,
        agent_poss=get_possessive(agent, True),
        obj=obj_mention,
        turn=turn,
        before_rel=rel_phrase(before_rel, agent),
        after_rel=rel_phrase(after_rel, agent),
        after_side=after_side,
        look_action=egocentric_look_action(after_rel),
        left_turn_rel=rel_phrase(left_turn_rel, agent),
        right_turn_rel=rel_phrase(right_turn_rel, agent),
    )
    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=f"implicit_turn_lr_contrast_{start_relation}_turn_{turn}",
        observer="a",
        before=before,
        after=after,
        text=text,
        difficulty=3,
        template_id=template_id,
        template_family="implicit_turn_lr_contrast",
    )


def generate_turn_lr_order_variant_item(
    rng: random.Random,
    *,
    agent: str,
    obj: str,
    example_id: int = 0,
    start_relation: str | None = None,
    turn: str | None = None,
    before_facing: str | None = None,
    template_variant: int | None = None,
) -> Item:
    """Generate front/back + turn examples with varied evidence order."""
    start_relation = start_relation or rng.choice(["front", "behind"])
    turn = turn or rng.choice(["left", "right"])
    before_facing = before_facing or rng.choice(HORIZONTAL)
    if start_relation not in {"front", "behind"}:
        raise ValueError(f"start_relation must be 'front' or 'behind', got {start_relation!r}")
    if turn not in {"left", "right"}:
        raise ValueError(f"turn must be 'left' or 'right', got {turn!r}")
    if before_facing not in HORIZONTAL:
        raise ValueError(f"before_facing must be one of {HORIZONTAL}, got {before_facing!r}")

    initial_delta = WORLD_DIRS[before_facing] if start_relation == "front" else neg(WORLD_DIRS[before_facing])
    before = State((0, 0, 0), initial_delta, before_facing, rng.choice(HORIZONTAL))
    after = State(before.a_pos, before.b_pos, turn_facing(before_facing, turn), before.b_facing)
    before_rel = compute_relations(before, "a")[1]
    after_rel = compute_relations(after, "a")[1]

    left_after = State(before.a_pos, before.b_pos, turn_facing(before_facing, "left"), before.b_facing)
    right_after = State(before.a_pos, before.b_pos, turn_facing(before_facing, "right"), before.b_facing)
    left_turn_rel = compute_relations(left_after, "a")[1]
    right_turn_rel = compute_relations(right_after, "a")[1]

    # v13/v15 test whether ordering is part of the bottleneck. The same spatial
    # composition appears with the premise first, the turn first, or the answer
    # first, because "pivot right + in front -> left-hand side" may generalize
    # differently from "in front + pivot right -> left-hand side".
    templates = [
        (
            "{obj} started {before_rel}. {agent} turned {turn} in place while "
            "{obj} stayed fixed. From the new stance, {obj} was {after_rel}."
        ),
        (
            "{agent} turned {turn} without stepping away. {obj} had been waiting "
            "{before_rel}. After the turn, {obj} was {after_rel}."
        ),
        (
            "From the new stance, {obj} was {after_rel}. That happened because "
            "{agent} turned {turn} while {obj} stayed where it had been, {before_rel}."
        ),
        (
            "When the fixed spot began {before_rel}, the pivot direction decided "
            "the side: left turn made it {left_turn_rel}, and right turn made it "
            "{right_turn_rel}."
        ),
    ]
    if template_variant is None:
        template_variant = rng.randrange(len(templates))
    if not 0 <= template_variant < len(templates):
        raise ValueError(f"template_variant must be in [0, {len(templates) - 1}], got {template_variant}")
    obj_mention = get_mention(obj, False)
    text = templates[template_variant].format(
        agent=agent,
        obj=obj_mention,
        turn=turn,
        before_rel=rel_phrase(before_rel, agent),
        after_rel=rel_phrase(after_rel, agent),
        left_turn_rel=rel_phrase(left_turn_rel, agent),
        right_turn_rel=rel_phrase(right_turn_rel, agent),
    )
    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=f"implicit_turn_lr_order_variants_{start_relation}_turn_{turn}",
        observer="a",
        before=before,
        after=after,
        text=text,
        difficulty=3,
        template_id=1100 + template_variant,
        template_family="implicit_turn_lr_order_variants",
    )


def generate_left_right_paired_contrast_item(
    rng: random.Random,
    *,
    agent: str,
    obj: str,
    example_id: int = 0,
    contrast_case: str | None = None,
    side: str | None = None,
    start_relation: str | None = None,
    before_facing: str | None = None,
    template_variant: int | None = None,
) -> Item:
    """Generate matched left/right pairs where one variable changes."""
    cases = ("facing_flip", "turn_direction_flip", "role_inversion")
    contrast_case = contrast_case or rng.choice(cases)
    side = side or rng.choice(["left", "right"])
    start_relation = start_relation or rng.choice(["front", "behind"])
    before_facing = before_facing or rng.choice(HORIZONTAL)
    if contrast_case not in cases:
        raise ValueError(f"contrast_case must be one of {cases}, got {contrast_case!r}")
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if start_relation not in {"front", "behind"}:
        raise ValueError(f"start_relation must be 'front' or 'behind', got {start_relation!r}")
    if before_facing not in HORIZONTAL:
        raise ValueError(f"before_facing must be one of {HORIZONTAL}, got {before_facing!r}")

    obj_mention = get_mention(obj, False)
    if contrast_case == "facing_flip":
        world_dir = turn_facing(before_facing, side)
        after_facing = turn_facing(before_facing, "around")
        before = State((0, 0, 0), WORLD_DIRS[world_dir], before_facing, rng.choice(HORIZONTAL))
        after = State(before.a_pos, before.b_pos, after_facing, before.b_facing)
        before_rel = compute_relations(before, "a")[1]
        after_rel = compute_relations(after, "a")[1]
        templates = [
            (
                "{agent} checked the same fixed spot twice. Facing {before_facing}, "
                "{obj} was {before_rel}. Facing {after_facing}, with nobody moving, "
                "{obj} was {after_rel}."
            ),
            (
                "The spot stayed {world_pos}. From a {before_facing}-facing stance it "
                "landed {before_rel}; from a {after_facing}-facing stance it landed "
                "{after_rel}."
            ),
        ]
        case_idx = 0
        format_args = {
            "agent": agent,
            "obj": obj_mention,
            "before_facing": before_facing,
            "after_facing": after_facing,
            "before_rel": rel_phrase(before_rel, agent),
            "after_rel": rel_phrase(after_rel, agent),
            "world_pos": world_phrase(world_relation(WORLD_DIRS[world_dir]), agent),
        }
        operation = f"implicit_left_right_paired_contrast_facing_flip_{side}"
    elif contrast_case == "turn_direction_flip":
        initial_delta = WORLD_DIRS[before_facing] if start_relation == "front" else neg(WORLD_DIRS[before_facing])
        before = State((0, 0, 0), initial_delta, before_facing, rng.choice(HORIZONTAL))
        left_after = State(before.a_pos, before.b_pos, turn_facing(before_facing, "left"), before.b_facing)
        right_after = State(before.a_pos, before.b_pos, turn_facing(before_facing, "right"), before.b_facing)
        left_turn_rel = compute_relations(left_after, "a")[1]
        right_turn_rel = compute_relations(right_after, "a")[1]
        before_rel = compute_relations(before, "a")[1]
        after = left_after
        templates = [
            (
                "{obj} stayed {before_rel}. If {agent} turned left, that fixed spot "
                "would be {left_turn_rel}. If {agent} turned right, the same spot "
                "would be {right_turn_rel}."
            ),
            (
                "Only the turn direction changed. With {obj} still {before_rel}, a "
                "left turn put it {left_turn_rel}; a right turn put it {right_turn_rel}."
            ),
        ]
        case_idx = 1
        format_args = {
            "agent": agent,
            "obj": obj_mention,
            "before_rel": rel_phrase(before_rel, agent),
            "left_turn_rel": rel_phrase(left_turn_rel, agent),
            "right_turn_rel": rel_phrase(right_turn_rel, agent),
        }
        operation = f"implicit_left_right_paired_contrast_turn_direction_{start_relation}"
    else:
        world_dir = turn_facing(before_facing, side)
        before = State((0, 0, 0), WORLD_DIRS[world_dir], before_facing, before_facing)
        after = before
        inverse_side = inverse_relation(side)
        templates = [
            (
                "{obj} was {obj_rel}. Looking at the same pair the other way, "
                "{agent} was {agent_rel}."
            ),
            (
                "The relation flipped when the reference point changed: {obj} was "
                "{obj_rel}, so {agent} was {agent_rel}."
            ),
        ]
        case_idx = 2
        format_args = {
            "agent": agent,
            "obj": obj_mention,
            "obj_rel": rel_phrase(side, agent),
            "agent_rel": rel_phrase(inverse_side, obj_mention),
        }
        operation = f"implicit_left_right_paired_contrast_role_inversion_{side}"

    # v19-v21: these are matched pairs by design. Unlike ordinary template
    # variety, each example holds most of the scene fixed while flipping one
    # variable: facing, turn direction, or subject/reference role.
    if template_variant is None:
        template_variant = rng.randrange(len(templates))
    if not 0 <= template_variant < len(templates):
        raise ValueError(f"template_variant must be in [0, {len(templates) - 1}], got {template_variant}")
    text = templates[template_variant].format(**format_args)
    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=operation,
        observer="a",
        before=before,
        after=after,
        text=text,
        difficulty=3,
        template_id=1200 + case_idx * 10 + template_variant,
        template_family="implicit_left_right_paired_contrast",
    )


def generate_distance_contrast_item(
    rng: random.Random,
    *,
    agent: str,
    obj: str,
    example_id: int = 0,
    distance_case: str | None = None,
    relation_view: str | None = None,
    direction: str | None = None,
    before_facing: str | None = None,
) -> Item:
    """Generate implicit close/far examples from reachability, gaps, or step count."""
    # v7 targets close/far separately because EWoK distance items often use
    # everyday reachability/contact semantics rather than pure coordinate
    # direction. The goal is to teach distance from touch, step count, gaps, and
    # toward/away changes, with direct and inverse relation wording.
    cases = ("reach_close", "reach_far", "compare_near", "compare_far", "move_toward", "move_away")
    distance_case = distance_case or rng.choice(cases)
    relation_view = relation_view or rng.choice(["direct", "inverse"])
    direction = direction or rng.choice(HORIZONTAL)
    before_facing = before_facing or rng.choice(HORIZONTAL)
    if distance_case not in cases:
        raise ValueError(f"distance_case must be one of {cases}, got {distance_case!r}")
    if relation_view not in {"direct", "inverse"}:
        raise ValueError(f"relation_view must be 'direct' or 'inverse', got {relation_view!r}")
    if direction not in HORIZONTAL:
        raise ValueError(f"direction must be one of {HORIZONTAL}, got {direction!r}")
    if before_facing not in HORIZONTAL:
        raise ValueError(f"before_facing must be one of {HORIZONTAL}, got {before_facing!r}")

    near_delta = WORLD_DIRS[direction]
    far_delta = scale(near_delta, rng.choice([3, 4]))
    obj_mention = get_mention(obj, False)
    agent_poss = get_possessive(agent, True)
    other_obj = rng.choice([candidate for candidate in OBJECTS if candidate != obj])
    other_mention = get_mention(other_obj, False)
    template_id = 700 + cases.index(distance_case)

    before = State((0, 0, 0), near_delta, before_facing, rng.choice(HORIZONTAL))
    after = before
    difficulty = 2

    if distance_case == "reach_close":
        subject, reference = (obj_mention, agent) if relation_view == "direct" else (agent, obj_mention)
        text = (
            f"{agent} could reach {obj_mention} from the same spot, without taking a step. "
            f"{subject} was close to {reference}."
        )
    elif distance_case == "reach_far":
        before = State((0, 0, 0), far_delta, before_facing, rng.choice(HORIZONTAL))
        after = before
        subject, reference = (obj_mention, agent) if relation_view == "direct" else (agent, obj_mention)
        text = (
            f"{agent} would need several steps across the room before reaching {obj_mention}. "
            f"{subject} was far from {reference}."
        )
    elif distance_case == "compare_near":
        difficulty = 3
        if relation_view == "direct":
            final_sentence = f"{obj_mention} was closer to {agent} than {other_mention}."
        else:
            final_sentence = f"{other_mention} was farther from {agent} than {obj_mention}."
        text = (
            f"{obj_mention} rested by {agent_poss} spot, while {other_mention} sat several steps away. "
            f"{final_sentence}"
        )
    elif distance_case == "compare_far":
        difficulty = 3
        before = State((0, 0, 0), far_delta, before_facing, rng.choice(HORIZONTAL))
        after = before
        if relation_view == "direct":
            final_sentence = f"{obj_mention} was farther from {agent} than {other_mention}."
        else:
            final_sentence = f"{other_mention} was closer to {agent} than {obj_mention}."
        text = (
            f"{obj_mention} sat several steps away from {agent}, while {other_mention} rested by "
            f"{agent_poss} spot. {final_sentence}"
        )
    elif distance_case == "move_toward":
        before = State((0, 0, 0), far_delta, before_facing, rng.choice(HORIZONTAL))
        after = State((0, 0, 0), near_delta, before.a_facing, before.b_facing)
        difficulty = 3
        final_sentence = (
            f"{obj_mention} was closer to {agent} than before."
            if relation_view == "direct"
            else f"{agent} was closer to {obj_mention} than before."
        )
        text = f"{agent} walked toward {obj_mention} and stopped near where it sat. {final_sentence}"
    else:
        before = State((0, 0, 0), near_delta, before_facing, rng.choice(HORIZONTAL))
        after = State((0, 0, 0), far_delta, before.a_facing, before.b_facing)
        difficulty = 3
        final_sentence = (
            f"{obj_mention} was farther from {agent} than before."
            if relation_view == "direct"
            else f"{agent} was farther from {obj_mention} than before."
        )
        text = f"{agent} moved away from {obj_mention} along the path. {final_sentence}"

    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=f"implicit_distance_contrast_{distance_case}_{relation_view}",
        observer="a",
        before=before,
        after=after,
        text=text,
        difficulty=difficulty,
        template_id=template_id,
        template_family="implicit_distance_contrast",
    )


def generate_distance_reciprocal_item(
    rng: random.Random,
    *,
    agent: str,
    obj: str,
    example_id: int = 0,
    relation: str | None = None,
    direction: str | None = None,
) -> Item:
    """Generate compact reciprocal close/far examples."""
    relation = relation or rng.choice(["close", "far"])
    direction = direction or rng.choice(HORIZONTAL)
    if relation not in {"close", "far"}:
        raise ValueError(f"relation must be 'close' or 'far', got {relation!r}")
    if direction not in HORIZONTAL:
        raise ValueError(f"direction must be one of {HORIZONTAL}, got {direction!r}")

    delta = WORLD_DIRS[direction] if relation == "close" else scale(WORLD_DIRS[direction], rng.choice([3, 4]))
    before = State((0, 0, 0), delta, rng.choice(HORIZONTAL), rng.choice(HORIZONTAL))
    obj_mention = get_mention(obj, False)
    if relation == "close":
        templates = [
            (
                "{agent} and {obj} were near enough to share the same small area. "
                "{obj} was close to {agent}, and {agent} was close to {obj}."
            ),
            (
                "{agent} stood beside {obj}. From either side of the pair, the other "
                "one was close."
            ),
        ]
    else:
        templates = [
            (
                "{agent} and {obj} were separated by several steps. {obj} was far "
                "from {agent}, and {agent} was far from {obj}."
            ),
            (
                "{agent} and {obj} stayed at opposite ends of the path. From either "
                "side of the pair, the other one was far away."
            ),
        ]
    template_variant = rng.randrange(len(templates))
    text = templates[template_variant].format(agent=agent, obj=obj_mention)
    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=f"implicit_distance_reciprocal_{relation}",
        observer="a",
        before=before,
        after=before,
        text=text,
        difficulty=3,
        template_id=760 + (0 if relation == "close" else 2) + template_variant,
        template_family="implicit_distance_reciprocal",
    )


def generate_turn_around_lr_item(
    rng: random.Random,
    *,
    agent: str,
    obj: str,
    example_id: int = 0,
    side: str | None = None,
    before_facing: str | None = None,
) -> Item:
    """Generate direct side-flip examples for an in-place turn around."""
    side = side or rng.choice(["left", "right"])
    before_facing = before_facing or rng.choice(HORIZONTAL)
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if before_facing not in HORIZONTAL:
        raise ValueError(f"before_facing must be one of {HORIZONTAL}, got {before_facing!r}")

    world_dir = turn_facing(before_facing, side)
    before = State((0, 0, 0), WORLD_DIRS[world_dir], before_facing, rng.choice(HORIZONTAL))
    after = State(before.a_pos, before.b_pos, turn_facing(before_facing, "around"), before.b_facing)
    before_rel = compute_relations(before, "a")[1]
    after_rel = compute_relations(after, "a")[1]
    obj_mention = get_mention(obj, False)
    templates = [
        (
            "{obj} was {before_rel} before {agent} turned around. After the turn, "
            "with no one changing places, {obj} was {after_rel}."
        ),
        (
            "{agent} spun to face the opposite way while {obj} stayed fixed. The "
            "spot that had been {before_rel} became {after_rel}."
        ),
        (
            "{obj} kept the same floor spot as {agent} reversed direction. From the "
            "new stance, {obj} was {after_rel}."
        ),
    ]
    template_variant = rng.randrange(len(templates))
    text = templates[template_variant].format(
        agent=agent,
        obj=obj_mention,
        before_rel=rel_phrase(before_rel, agent),
        after_rel=rel_phrase(after_rel, agent),
    )
    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=f"implicit_turn_around_lr_{side}",
        observer="a",
        before=before,
        after=after,
        text=text,
        difficulty=3,
        template_id=800 + template_variant,
        template_family="implicit_turn_around_lr",
    )


def generate_cardinal_guard_item(
    rng: random.Random,
    *,
    agent: str,
    obj: str,
    example_id: int = 0,
    case: str | None = None,
    relation: str | None = None,
) -> Item:
    """Generate world-direction examples that avoid egocentric left/right language."""
    case = case or rng.choice(["preserve", "pass"])
    relation = relation or rng.choice(HORIZONTAL)
    if case not in {"preserve", "pass"}:
        raise ValueError(f"case must be 'preserve' or 'pass', got {case!r}")
    if relation not in HORIZONTAL:
        raise ValueError(f"relation must be one of {HORIZONTAL}, got {relation!r}")

    step = WORLD_DIRS[relation]
    obj_mention = get_mention(obj, False)
    if case == "preserve":
        move_dir = rng.choice(HORIZONTAL)
        move = WORLD_DIRS[move_dir]
        before = State((0, 0, 0), step, rng.choice(HORIZONTAL), rng.choice(HORIZONTAL))
        after = State(add(before.a_pos, move), add(before.b_pos, move), before.a_facing, before.b_facing)
        text = (
            f"{obj_mention} started {world_phrase(relation, agent)}. {agent} and {obj_mention} "
            f"both shifted one step {move_dir}. {obj_mention} stayed {world_phrase(relation, agent)}."
        )
        template_id = 900
    else:
        before = State((0, 0, 0), step, rng.choice(HORIZONTAL), rng.choice(HORIZONTAL))
        after = State(scale(step, 2), step, before.a_facing, before.b_facing)
        after_world = compute_relations(after, "a")[0]
        text = (
            f"{agent} walked {relation} past {obj_mention} and stopped beyond it. "
            f"{obj_mention} ended up {world_phrase(after_world, agent)}."
        )
        template_id = 901
    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=f"implicit_cardinal_guard_{case}_{relation}",
        observer="a",
        before=before,
        after=after,
        text=text,
        difficulty=2,
        template_id=template_id,
        template_family="implicit_cardinal_guard",
    )


def generate_cardinal_inverse_item(
    rng: random.Random,
    *,
    agent: str,
    obj: str,
    example_id: int = 0,
    relation: str | None = None,
    template_variant: int | None = None,
) -> Item:
    """Generate direct plus reciprocal cardinal-direction facts."""
    relation = relation or rng.choice(CARDINAL)
    if relation not in CARDINAL:
        raise ValueError(f"relation must be one of {CARDINAL}, got {relation!r}")

    inverse = inverse_relation(relation)
    before = State((0, 0, 0), WORLD_DIRS[relation], rng.choice(HORIZONTAL), rng.choice(HORIZONTAL))
    obj_mention = get_mention(obj, False)
    templates = [
        (
            "{obj} was marked {rel_agent}. Reading the same map from {obj_poss} spot, "
            "{agent} was {inv_obj}."
        ),
        (
            "A note placed {obj} {rel_agent}. The reverse note is therefore that "
            "{agent} is {inv_obj}."
        ),
        (
            "On the sketch, {obj} sat {rel_agent}. From {obj}, the matching return "
            "direction put {agent} {inv_obj}."
        ),
    ]
    if template_variant is None:
        template_variant = rng.randrange(len(templates))
    if not 0 <= template_variant < len(templates):
        raise ValueError(f"template_variant must be in [0, {len(templates) - 1}], got {template_variant}")
    text = templates[template_variant].format(
        agent=agent,
        obj=obj_mention,
        obj_poss=get_possessive(obj, False),
        rel_agent=world_phrase(relation, agent),
        inv_obj=world_phrase(inverse, obj_mention),
    )
    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=f"implicit_cardinal_inverse_{relation}",
        observer="a",
        before=before,
        after=before,
        text=text,
        difficulty=1,
        template_id=1100 + template_variant,
        template_family="implicit_cardinal_inverse",
    )


def generate_cardinal_move_past_item(
    rng: random.Random,
    *,
    agent: str,
    obj: str,
    example_id: int = 0,
    relation: str | None = None,
    template_variant: int | None = None,
) -> Item:
    """Generate straight cardinal motion that crosses a fixed landmark."""
    relation = relation or rng.choice(CARDINAL)
    if relation not in CARDINAL:
        raise ValueError(f"relation must be one of {CARDINAL}, got {relation!r}")

    step = WORLD_DIRS[relation]
    before = State((0, 0, 0), step, relation, rng.choice(HORIZONTAL))
    after = State(scale(step, 2), step, relation, before.b_facing)
    final_rel = compute_relations(after, "a")[0]
    obj_mention = get_mention(obj, False)
    templates = [
        (
            "{agent} walked {relation} toward {obj} and continued past it. "
            "After stopping, {obj} was {final_rel}."
        ),
        (
            "{obj} stayed fixed as {agent} traveled {relation} beyond it. "
            "At the end of the route, {obj} was {final_rel}."
        ),
        (
            "{agent} crossed the spot by {obj} while moving {relation}. "
            "Once {agent} was beyond that spot, {obj} lay {final_rel}."
        ),
    ]
    if template_variant is None:
        template_variant = rng.randrange(len(templates))
    if not 0 <= template_variant < len(templates):
        raise ValueError(f"template_variant must be in [0, {len(templates) - 1}], got {template_variant}")
    text = templates[template_variant].format(
        agent=agent,
        obj=obj_mention,
        relation=relation,
        final_rel=world_phrase(final_rel, agent),
    )
    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=f"implicit_cardinal_move_past_{relation}",
        observer="a",
        before=before,
        after=after,
        text=text,
        difficulty=3,
        template_id=1110 + template_variant,
        template_family="implicit_cardinal_move_past",
    )


def generate_cardinal_landmark_item(
    rng: random.Random,
    *,
    agent: str,
    obj: str,
    example_id: int = 0,
    relation: str | None = None,
    template_variant: int | None = None,
) -> Item:
    """Generate map/landmark wording for direct and inverse cardinal facts."""
    relation = relation or rng.choice(CARDINAL)
    if relation not in CARDINAL:
        raise ValueError(f"relation must be one of {CARDINAL}, got {relation!r}")

    inverse = inverse_relation(relation)
    before = State((0, 0, 0), WORLD_DIRS[relation], rng.choice(HORIZONTAL), rng.choice(HORIZONTAL))
    obj_mention = get_mention(obj, False)
    templates = [
        (
            "The map placed {obj} {rel_agent} from {agent}'s marker. "
            "That same layout placed {agent}'s marker {inv_obj}."
        ),
        (
            "Using {agent}'s marker as the reference, {obj} was {rel_agent}. "
            "Using {obj} as the reference, {agent}'s marker was {inv_obj}."
        ),
        (
            "A route card listed {obj} {rel_agent}. The return line on the card "
            "listed {agent}'s marker {inv_obj}."
        ),
    ]
    if template_variant is None:
        template_variant = rng.randrange(len(templates))
    if not 0 <= template_variant < len(templates):
        raise ValueError(f"template_variant must be in [0, {len(templates) - 1}], got {template_variant}")
    text = templates[template_variant].format(
        agent=agent,
        obj=obj_mention,
        rel_agent=world_phrase(relation, agent),
        inv_obj=world_phrase(inverse, obj_mention),
    )
    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=f"implicit_cardinal_landmark_{relation}",
        observer="a",
        before=before,
        after=before,
        text=text,
        difficulty=2,
        template_id=1120 + template_variant,
        template_family="implicit_cardinal_landmark",
    )


def generate_cardinal_route_update_item(
    rng: random.Random,
    *,
    agent: str,
    obj: str,
    example_id: int = 0,
    relation: str | None = None,
    template_variant: int | None = None,
) -> Item:
    """Generate cardinal relations that stay fixed under shared translation."""
    relation = relation or rng.choice(CARDINAL)
    if relation not in CARDINAL:
        raise ValueError(f"relation must be one of {CARDINAL}, got {relation!r}")

    move_dir = rng.choice(CARDINAL)
    before = State((0, 0, 0), WORLD_DIRS[relation], rng.choice(HORIZONTAL), rng.choice(HORIZONTAL))
    move = WORLD_DIRS[move_dir]
    after = State(add(before.a_pos, move), add(before.b_pos, move), before.a_facing, before.b_facing)
    obj_mention = get_mention(obj, False)
    templates = [
        (
            "{obj} began {rel_agent}. Then {agent} and {obj} both shifted {move_dir} "
            "by the same amount. The shift kept {obj} {rel_agent}."
        ),
        (
            "Before the route changed, {obj} was {rel_agent}. {agent} and {obj} "
            "moved together toward the {move_dir}. Their spacing stayed the same, "
            "so {obj} remained {rel_agent}."
        ),
        (
            "{agent} and {obj} slid together in the {move_dir} direction. Since both "
            "moved equally, {obj} was still {rel_agent} afterward."
        ),
    ]
    if template_variant is None:
        template_variant = rng.randrange(len(templates))
    if not 0 <= template_variant < len(templates):
        raise ValueError(f"template_variant must be in [0, {len(templates) - 1}], got {template_variant}")
    text = templates[template_variant].format(
        agent=agent,
        obj=obj_mention,
        rel_agent=world_phrase(relation, agent),
        move_dir=move_dir,
    )
    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=f"implicit_cardinal_route_update_{relation}_move_{move_dir}",
        observer="a",
        before=before,
        after=after,
        text=text,
        difficulty=2,
        template_id=1130 + template_variant,
        template_family="implicit_cardinal_route_update",
    )


def generate_relation_type_contrast_item(
    rng: random.Random,
    *,
    agent: str,
    obj: str,
    example_id: int = 0,
    case: str | None = None,
) -> Item:
    """Generate examples contrasting symmetric and inverse spatial relations."""
    cases = ("symmetric_distance", "inverse_cardinal", "inverse_vertical", "inverse_left_right")
    case = case or rng.choice(cases)
    if case not in cases:
        raise ValueError(f"case must be one of {cases}, got {case!r}")

    obj_mention = get_mention(obj, False)
    if case == "symmetric_distance":
        direction = rng.choice(HORIZONTAL)
        before = State((0, 0, 0), WORLD_DIRS[direction], rng.choice(HORIZONTAL), rng.choice(HORIZONTAL))
        text = (
            f"{agent} and {obj_mention} stood close together. From {agent}'s spot, "
            f"{obj_mention} was nearby; from {obj_mention}'s spot, {agent} was nearby too."
        )
        observer = "a"
    elif case == "inverse_cardinal":
        relation = rng.choice(HORIZONTAL)
        before = State((0, 0, 0), WORLD_DIRS[relation], rng.choice(HORIZONTAL), rng.choice(HORIZONTAL))
        inverse = inverse_relation(relation)
        text = (
            f"{obj_mention} sat {world_phrase(relation, agent)}. Looking back from "
            f"{obj_mention}, {agent} was {world_phrase(inverse, obj_mention)}."
        )
        observer = "a"
    elif case == "inverse_vertical":
        relation = rng.choice(["above", "below"])
        before = State((0, 0, 0), WORLD_DIRS[relation], rng.choice(HORIZONTAL), rng.choice(HORIZONTAL))
        inverse = inverse_relation(relation)
        text = (
            f"{obj_mention} was {world_phrase(relation, agent)}. From {obj_mention}'s "
            f"height, {agent} was {world_phrase(inverse, obj_mention)}."
        )
        observer = "a"
    else:
        facing = rng.choice(HORIZONTAL)
        side = rng.choice(["left", "right"])
        a_pos = WORLD_DIRS[turn_facing(facing, side)]
        before = State(a_pos, (0, 0, 0), facing, facing)
        reciprocal_rel = compute_relations(before, "a")[1]
        text = (
            f"{agent} and {obj_mention} faced {facing}. {agent} stood {rel_phrase(side, obj_mention)}. "
            f"From {agent}'s side, {obj_mention} was {rel_phrase(reciprocal_rel, agent)}."
        )
        observer = "a"

    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=f"implicit_relation_type_contrast_{case}",
        observer=observer,
        before=before,
        after=before,
        text=text,
        difficulty=3,
        template_id=1000 + cases.index(case),
        template_family="implicit_relation_type_contrast",
    )


def generate_pass_by_implicit_item(
    rng: random.Random,
    *,
    agent: str,
    obj: str,
    example_id: int = 0,
) -> Item:
    """Generate a pass-by scene that flips front into behind without rule wording."""
    a_facing = rng.choice(HORIZONTAL)
    b_facing = rng.choice(HORIZONTAL)
    step = WORLD_DIRS[a_facing]
    before = State((0, 0, 0), step, a_facing, b_facing)
    after = State(scale(step, 2), step, a_facing, b_facing)
    after_rel = compute_relations(after, "a")[1]
    template_id = 300 + rng.randrange(3)
    templates = [
        (
            "{agent} moved toward {obj}, went past it, and kept going without turning. "
            "By the time {agent} stopped, {obj} was {after_rel}."
        ),
        (
            "{agent} headed straight toward {obj} and continued beyond it. After that "
            "short walk, {obj} was {after_rel}."
        ),
        (
            "{obj} stayed put as {agent} walked straight past. When the motion was over, "
            "{obj} was {after_rel}."
        ),
    ]
    text = templates[template_id - 300].format(
        agent=agent,
        obj=get_mention(obj, False),
        after_rel=rel_phrase(after_rel, agent),
    )
    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation="implicit_pass_by_agent_forward",
        observer="a",
        before=before,
        after=after,
        text=text,
        difficulty=2,
        template_id=template_id,
        template_family="implicit_pass_by",
    )


def _pass_through_final_sentence(agent: str, obj: str, after: State, relation_view: str) -> Tuple[str, str]:
    obj_mention = get_mention(obj, False)
    if relation_view == "object_egocentric":
        _, after_rel = compute_relations(after, "a")
        return f"{obj_mention} was {rel_phrase(after_rel, agent)}", "a"
    if relation_view == "object_world":
        after_world, _ = compute_relations(after, "a")
        return f"{obj_mention} was {world_phrase(after_world, agent)}", "a"
    if relation_view == "agent_world":
        after_world, _ = compute_relations(after, "b")
        return f"{agent} was {world_phrase(after_world, obj_mention)}", "b"
    raise ValueError(f"Unknown pass-through relation_view={relation_view!r}")


def generate_pass_through_implicit_item(
    rng: random.Random,
    *,
    agent: str,
    obj: str,
    example_id: int = 0,
    relation_view: str | None = None,
    template_variant: int | None = None,
) -> Item:
    """Generate straight-line pass-through scenes with varied non-EWoK wording."""
    travel_dir = rng.choice(HORIZONTAL)
    step = WORLD_DIRS[travel_dir]
    b_facing = rng.choice(HORIZONTAL)
    before = State((0, 0, 0), step, travel_dir, b_facing)
    after = State(scale(step, 2), step, travel_dir, b_facing)

    relation_view = relation_view or rng.choice(["object_egocentric", "object_world", "agent_world"])
    final_sentence, observer = _pass_through_final_sentence(agent, obj, after, relation_view)
    obj_mention = get_mention(obj, False)

    # Future pass-through ablations worth keeping separate:
    # - kept-going: "went by without stopping", "passed and carried on", "moved past without turning back"
    # - crossed/overshot: "crossed to the far side", "went farther along the same path", "overshot on a straight route"
    # - landmark: "used it as a point on the path", "crossed the middle marker", "passed the marker"
    # - direction-specific: "kept moving north/east/south/west", "crossed its place", "walked from one side to the other"
    # - before/after relation: "was ahead at first", "began before it", "moved from one side to the other"
    # - inverse wording: "finished on the far side", "swapped sides along the path", "left it on the opposite side"
    # - everyday scenes: "walked down a hallway", "moved along a road", "ran along a path"
    templates = [
        (
            "{agent} went by {obj} without stopping. {agent} kept the same heading, "
            "and when the walk ended, {final_sentence}."
        ),
        (
            "{agent} passed {obj} and carried on a little farther. By the end, "
            "{final_sentence}."
        ),
        (
            "{agent} moved past {obj} without turning back. After the movement, "
            "{final_sentence}."
        ),
        (
            "{agent} crossed the spot beside {obj} and stopped on the far side. "
            "Afterward, {final_sentence}."
        ),
        (
            "{agent} went farther than {obj} along the same path. When {agent} stopped, "
            "{final_sentence}."
        ),
        (
            "{agent} overshot {obj} on a straight route. At the finish, {final_sentence}."
        ),
        (
            "{agent} used {obj} as a point on the path, went by it, and stopped beyond it. "
            "Then {final_sentence}."
        ),
        (
            "{obj} marked the middle of {agent_poss} route. {agent} crossed that mark and "
            "stopped past it. In the final layout, {final_sentence}."
        ),
        (
            "{agent} passed the marker made by {obj} and kept the same direction. "
            "At the end, {final_sentence}."
        ),
        (
            "{agent} moved {travel_dir} toward {obj}, then kept moving {travel_dir} past it. "
            "Afterward, {final_sentence}."
        ),
        (
            "{agent} followed a {travel_dir}-going line that crossed {obj_poss} place. "
            "When the movement ended, {final_sentence}."
        ),
        (
            "{agent} started on one side of {obj} and walked {travel_dir} to the other side. "
            "After that, {final_sentence}."
        ),
        (
            "{obj} was ahead of {agent} at first. {agent} went by without turning. "
            "At the end, {final_sentence}."
        ),
        (
            "{agent} began before {obj} on the path and finished after passing it. "
            "In the new arrangement, {final_sentence}."
        ),
        (
            "{agent} moved from one side of {obj} to the other. Once the move was over, "
            "{final_sentence}."
        ),
        (
            "{agent} finished on the far side of {obj}. From the final positions, "
            "{final_sentence}."
        ),
        (
            "After {agent} crossed past {obj}, the two had swapped sides along the path. "
            "{final_sentence}."
        ),
        (
            "{agent} left {obj} on the opposite side from where the walk began. "
            "In the final layout, {final_sentence}."
        ),
        (
            "{agent} walked down a hallway past {obj}. When {agent} stopped, "
            "{final_sentence}."
        ),
        (
            "{agent} moved along a road past {obj}. After the short trip, "
            "{final_sentence}."
        ),
        (
            "{agent} ran along the path past {obj} and stopped beyond it. At the end, "
            "{final_sentence}."
        ),
    ]
    if template_variant is None:
        template_variant = rng.randrange(len(templates))
    if not 0 <= template_variant < len(templates):
        raise ValueError(f"template_variant must be in [0, {len(templates) - 1}], got {template_variant}")
    template_id = 400 + template_variant
    text = templates[template_variant].format(
        agent=agent,
        agent_poss=get_possessive(agent, True),
        obj=obj_mention,
        obj_poss=get_possessive(obj, False),
        travel_dir=travel_dir,
        final_sentence=final_sentence,
    )
    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=f"implicit_pass_through_agent_{travel_dir}_{relation_view}",
        observer=observer,
        before=before,
        after=after,
        text=text,
        difficulty=2,
        template_id=template_id,
        template_family="implicit_pass_through",
    )


def validate_item(item: Item) -> None:
    before_world, before_rel = compute_relations(item.before_state, item.observer)
    after_world, after_rel = compute_relations(item.after_state, item.observer)
    assert before_world == item.before_world and before_rel == item.before_relative
    assert after_world == item.after_world and after_rel == item.after_relative

    for agent in AGENTS:
        assert f"the {agent}" not in item.text
        assert f"the {agent}'s" not in item.text
    for obj in OBJECTS:
        assert f"of {obj}" not in item.text.replace(f"of the {obj}", "")


def generate_item(rng: random.Random, *, example_id: int = 0, template_preset: str = "v4") -> Item:
    agent = rng.choice(AGENTS)
    obj = rng.choice(OBJECTS)

    try:
        families, weights = TEMPLATE_PRESETS[template_preset]
    except KeyError as exc:
        valid = ", ".join(sorted(TEMPLATE_PRESETS))
        raise ValueError(f"Unsupported template_preset={template_preset!r}; expected one of: {valid}") from exc
    template_family = rng.choices(families, weights=weights, k=1)[0]

    if template_family == "implicit_vertical":
        return generate_vertical_implicit_item(rng, agent=agent, obj=obj, example_id=example_id)
    if template_family == "implicit_turn":
        return generate_turn_implicit_item(rng, agent=agent, obj=obj, example_id=example_id)
    if template_family == "implicit_left_right_contrast":
        return generate_left_right_contrast_item(rng, agent=agent, obj=obj, example_id=example_id)
    if template_family == "implicit_turn_lr_contrast":
        return generate_turn_lr_contrast_item(rng, agent=agent, obj=obj, example_id=example_id)
    if template_family == "implicit_turn_lr_order_variants":
        return generate_turn_lr_order_variant_item(rng, agent=agent, obj=obj, example_id=example_id)
    if template_family == "implicit_left_right_paired_contrast":
        return generate_left_right_paired_contrast_item(rng, agent=agent, obj=obj, example_id=example_id)
    if template_family == "implicit_distance_contrast":
        return generate_distance_contrast_item(rng, agent=agent, obj=obj, example_id=example_id)
    if template_family == "implicit_distance_reciprocal":
        return generate_distance_reciprocal_item(rng, agent=agent, obj=obj, example_id=example_id)
    if template_family == "implicit_turn_around_lr":
        return generate_turn_around_lr_item(rng, agent=agent, obj=obj, example_id=example_id)
    if template_family == "implicit_cardinal_guard":
        return generate_cardinal_guard_item(rng, agent=agent, obj=obj, example_id=example_id)
    if template_family == "implicit_cardinal_inverse":
        return generate_cardinal_inverse_item(rng, agent=agent, obj=obj, example_id=example_id)
    if template_family == "implicit_cardinal_move_past":
        return generate_cardinal_move_past_item(rng, agent=agent, obj=obj, example_id=example_id)
    if template_family == "implicit_cardinal_landmark":
        return generate_cardinal_landmark_item(rng, agent=agent, obj=obj, example_id=example_id)
    if template_family == "implicit_cardinal_route_update":
        return generate_cardinal_route_update_item(rng, agent=agent, obj=obj, example_id=example_id)
    if template_family == "implicit_relation_type_contrast":
        return generate_relation_type_contrast_item(rng, agent=agent, obj=obj, example_id=example_id)
    if template_family == "implicit_pass_by":
        return generate_pass_by_implicit_item(rng, agent=agent, obj=obj, example_id=example_id)
    if template_family == "implicit_pass_through":
        return generate_pass_through_implicit_item(rng, agent=agent, obj=obj, example_id=example_id)

    a_pos = (0, 0, 0)
    a_facing = rng.choice(HORIZONTAL)
    b_facing = rng.choice(HORIZONTAL)
    initial_delta = rng.choice(list(WORLD_DIRS.values()))
    b_pos = add(a_pos, initial_delta)
    observer = rng.choice(["a", "b"])

    before = State(a_pos, b_pos, a_facing, b_facing)
    after, operation = apply_operation(before, rng)

    text, difficulty, template_id = render_text(agent, obj, before, after, operation, observer, rng)
    return _build_item(
        example_id=example_id,
        agent=agent,
        obj=obj,
        operation=operation,
        observer=observer,
        before=before,
        after=after,
        text=text,
        difficulty=difficulty,
        template_id=template_id,
        template_family="explicit",
    )


def _difficulty_targets(n: int, difficulty: str) -> Dict[str, int] | None:
    if difficulty != "mixed":
        return {difficulty: n}
    base = n // len(DIFFICULTY_LABELS)
    remainder = n % len(DIFFICULTY_LABELS)
    return {
        label: base + (1 if idx < remainder else 0)
        for idx, label in enumerate(DIFFICULTY_LABELS)
    }


def generate_items(
    *,
    n: int,
    seed: int,
    difficulty: str = "mixed",
    template_preset: str = "v4",
    max_attempts_multiplier: int = 200,
) -> List[Item]:
    if n <= 0:
        return []
    if difficulty not in {"mixed", *DIFFICULTY_LABELS}:
        raise ValueError(f"Unsupported difficulty={difficulty!r}")
    if template_preset not in TEMPLATE_PRESETS:
        valid = ", ".join(sorted(TEMPLATE_PRESETS))
        raise ValueError(f"Unsupported template_preset={template_preset!r}; expected one of: {valid}")

    rng = random.Random(seed)
    targets = _difficulty_targets(n, difficulty)
    counts: Counter[str] = Counter()
    out: List[Item] = []
    attempts = 0
    max_attempts = max(n * max_attempts_multiplier, 1000)

    while len(out) < n and attempts < max_attempts:
        attempts += 1
        candidate = generate_item(rng, example_id=len(out), template_preset=template_preset)
        label = candidate.difficulty_label
        if targets is not None and counts[label] >= targets.get(label, 0):
            continue
        candidate = replace(candidate, example_id=len(out))
        out.append(candidate)
        counts[label] += 1

    if len(out) != n:
        raise RuntimeError(
            f"Could only generate {len(out)} / {n} examples after {attempts} attempts "
            f"for difficulty={difficulty!r}; counts={dict(counts)}"
        )
    return out


def _state_json(state: State) -> str:
    return json.dumps(asdict(state), sort_keys=True)


def split_context_completion(text: str) -> Tuple[str, str]:
    text = " ".join(str(text).strip().split())
    split_at = text.rfind(". ")
    if split_at < 0:
        return "", text
    context = text[: split_at + 1].strip()
    completion = text[split_at + 2 :].strip()
    return context, completion


def item_to_row(item: Item, *, seed: int) -> Dict[str, object]:
    context, completion = split_context_completion(item.text)
    return {
        "example_id": item.example_id,
        "text": item.text,
        "context": context,
        "completion": completion,
        "loss_start_char": len(context) + (1 if context and completion else 0),
        "difficulty": item.difficulty,
        "difficulty_label": item.difficulty_label,
        "domain": "spatial-relations",
        "agent": item.agent,
        "object": item.obj,
        "operation": item.operation,
        "observer": item.observer,
        "before_world": item.before_world,
        "before_relative": item.before_relative,
        "after_world": item.after_world,
        "after_relative": item.after_relative,
        "template_id": item.template_id,
        "template_family": item.template_family,
        "before_state": _state_json(item.before_state),
        "after_state": _state_json(item.after_state),
        "seed": seed,
    }


def write_csv(items: Iterable[Item], out_path: Path, *, seed: int) -> None:
    rows = [item_to_row(item, seed=seed) for item in items]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "example_id",
        "text",
        "context",
        "completion",
        "loss_start_char",
        "difficulty",
        "difficulty_label",
        "domain",
        "agent",
        "object",
        "operation",
        "observer",
        "before_world",
        "before_relative",
        "after_world",
        "after_relative",
        "template_id",
        "template_family",
        "before_state",
        "after_state",
        "seed",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(items: Iterable[Item], out_path: Path, *, seed: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item_to_row(item, seed=seed), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10000, help="Number of examples to generate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--difficulty",
        choices=("mixed", *DIFFICULTY_LABELS),
        default="mixed",
        help="mixed balances easy/medium/hard equally.",
    )
    parser.add_argument(
        "--template-preset",
        choices=tuple(TEMPLATE_PRESETS),
        default="v4",
        help=(
            "v3 excludes implicit_pass_through; v4 includes it; v5 adds left/right "
            "contrastive facing pairs; v6 adds front/back-to-left/right turn contrasts; "
            "v7 adds close/far distance contrasts; v8-v12 branch from v6 to test reciprocal distance, "
            "distance-lite, turn-around left/right, cardinal guardrails, and relation-type contrast; "
            "v13-v15 compose v8 with turn-order and turn-around left/right variants; "
            "v19-v21 add matched left/right contrast pairs at increasing density; "
            "cardinal_v1 generates only north/south and east/west text for natural-data mixing."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/research/bos_aligned_proto/spatial_synth/spatial_relations_synth.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--jsonl-out", type=Path, default=None, help="Optional mirror JSONL output path.")
    parser.add_argument("--preview", type=int, default=3, help="Print this many generated examples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = generate_items(
        n=args.n,
        seed=args.seed,
        difficulty=args.difficulty,
        template_preset=args.template_preset,
    )
    write_csv(items, args.out, seed=args.seed)
    if args.jsonl_out is not None:
        write_jsonl(items, args.jsonl_out, seed=args.seed)

    counts = Counter(item.difficulty_label for item in items)
    print(f"Wrote {len(items)} examples to {args.out}")
    print(f"Difficulty counts: {dict(counts)}")
    for item in items[: max(0, args.preview)]:
        print(f"[{item.example_id} | {item.difficulty_label}] {item.text}")


if __name__ == "__main__":
    main()
